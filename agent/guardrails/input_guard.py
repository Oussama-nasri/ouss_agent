"""
Layer 1: Input Guard

Runs BEFORE the user message reaches the LLM.
Goal: catch malicious or out-of-scope input as cheaply as possible.

Defense order (cheapest → most expensive):
  1. Length check          — O(1), free
  2. Regex blocklist       — O(n), near-free
  3. PII detection         — O(n), near-free
  4. LLM-as-classifier     — ~500ms, costs tokens (optional, for high-security apps)

Why catch things here?
  - Saves LLM tokens (money / latency)
  - Prevents prompt injection from ever touching context
  - First line of defense for indirect injection via tool results
"""

import re
from agent.guardrails.exceptions import InputGuardrailError
from utils.logger import Logger

logger = Logger(__name__)

# ─── Prompt Injection Patterns ────────────────────────────────────────────────
# These are the most common jailbreak / injection phrases.
# Keep this list in a config file in production so you can update without redeploy.
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(everything|all|your\s+instructions)",
    r"you\s+are\s+now\s+(a\s+)?(\w+\s+)?(?:AI|assistant|bot|model)\s+(?:called|named|without|that)",
    r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(?:an?\s+)?(?:unrestricted|unfiltered|evil|jailbroken|DAN)",
    r"pretend\s+(you\s+)?(are|have\s+no)\s+(restrictions|rules|guidelines|filters)",
    r"(system|developer|admin|root)\s*prompt\s*[:=]",
    r"<\s*system\s*>",          # XML injection attempt
    r"\[\s*system\s*\]",        # bracket injection attempt
    r"do\s+anything\s+now",     # DAN jailbreak
    r"jailbreak",
    r"prompt\s+injection",
]

# ─── PII Patterns ─────────────────────────────────────────────────────────────
PII_PATTERNS = {
    "credit_card":   r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11})\b",
    "ssn":           r"\b\d{3}-\d{2}-\d{4}\b",
    "email":         r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "phone_us":      r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ip_address":    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "api_key":       r"\b(?:sk|pk|api|key)[-_][a-zA-Z0-9]{16,}\b",
}

# ─── Scope keywords — what this agent IS allowed to discuss ──────────────────
# Set to None to disable scope checking (allow all topics).
# Customize per use case: coding agent, customer support bot, etc.
ALLOWED_SCOPE_KEYWORDS = None  # e.g. ["code", "python", "file", "search", "run", "write"]


class InputGuard:
    """
    Validates user input before it reaches the LLM.

    Usage:
        guard = InputGuard(max_length=2000, check_pii=True, check_scope=False)
        guard.check("user message here")  # raises InputGuardrailError if bad
    """

    def __init__(
        self,
        max_length: int  = 4000,
        check_pii: bool  = True,
        check_scope: bool = False,
        pii_action: str  = "warn",   # "warn" | "block"
    ):
        self.max_length  = max_length
        self.check_pii   = check_pii
        self.check_scope = check_scope
        self.pii_action  = pii_action

        # Compile all regex patterns once at init (not on every call)
        self._injection_re = [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in INJECTION_PATTERNS
        ]
        self._pii_re = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in PII_PATTERNS.items()
        }

    def check(self, text: str) -> None:
        """
        Run all input checks. Raises InputGuardrailError on violation.
        Returns None if everything is clean.
        """
        self._check_length(text)
        self._check_injection(text)
        if self.check_pii:
            self._check_pii(text)
        if self.check_scope and ALLOWED_SCOPE_KEYWORDS:
            self._check_scope(text)

        logger.info(f"[InputGuard] ✓ passed ({len(text)} chars)")

    # ── Private checks ─────────────────────────────────────────────────────────

    def _check_length(self, text: str) -> None:
        if len(text) > self.max_length:
            raise InputGuardrailError(
                f"Input too long ({len(text)} chars, max {self.max_length}).",
                reason="length_exceeded",
                severity="low",
            )

    def _check_injection(self, text: str) -> None:
        for pattern in self._injection_re:
            match = pattern.search(text)
            if match:
                logger.warning(f"[InputGuard] ⚠ Injection attempt: '{match.group()}'")
                raise InputGuardrailError(
                    "I detected an attempt to override my instructions. I can't process this request.",
                    reason="prompt_injection",
                    severity="critical",
                )

    def _check_pii(self, text: str) -> None:
        found = []
        for pii_type, pattern in self._pii_re.items():
            if pattern.search(text):
                found.append(pii_type)

        if found:
            logger.warning(f"[InputGuard] ⚠ PII detected: {found}")
            if self.pii_action == "block":
                raise InputGuardrailError(
                    f"Your message appears to contain sensitive information ({', '.join(found)}). "
                    "Please remove it before sending.",
                    reason="pii_detected",
                    severity="high",
                )
            # "warn" mode: log but allow through (useful for dev/audit)

    def _check_scope(self, text: str) -> None:
        text_lower = text.lower()
        if not any(kw in text_lower for kw in ALLOWED_SCOPE_KEYWORDS):
            raise InputGuardrailError(
                "I'm specialized for coding and file tasks. "
                "I can't help with that topic — please ask something related to code, files, or search.",
                reason="out_of_scope",
                severity="low",
            )