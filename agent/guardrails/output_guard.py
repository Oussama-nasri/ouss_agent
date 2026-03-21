"""
Layer 3: Output Guard

Runs AFTER the LLM responds, BEFORE the response reaches the user.
Goal: catch harmful, hallucinated, or policy-violating outputs.

Checks (in order):
  1. Length sanity         — detect runaway/empty responses
  2. Toxic content         — keyword + pattern based toxicity scoring
  3. Hallucination signals — fabricated tool results, impossible claims
  4. PII leakage           — did the model echo back sensitive data?
  5. Refusal bypass check  — did the model accidentally comply with a jailbreak?

State-of-the-art note:
  For production, replace the keyword toxicity scorer with a dedicated model:
    - LlamaGuard (Meta)       → runs locally via Ollama
    - Detoxify (HuggingFace)  → fast, lightweight Python package
    - OpenAI Moderation API   → if you're already using OpenAI
"""

import re
from agent.guardrails.exceptions import OutputGuardrailError
from utils.logger import Logger

logger = Logger(__name__)

# ─── Toxicity keywords (tiered by severity) ───────────────────────────────────
# In production: replace with a classifier model (Detoxify, LlamaGuard)
TOXIC_HIGH = [
    r"\b(kill|murder|assassinate)\s+(yourself|himself|herself|themselves|people)\b",
    r"\b(make|build|create|synthesize)\s+(a\s+)?(bomb|weapon|explosive|poison|malware|virus)\b",
    r"\b(child\s+)?(porn|pornography|sexual\s+abuse)\b",
    r"\bhow\s+to\s+(hack|exploit|attack)\s+\w+\s+(system|server|network|database)\b",
]

TOXIC_MEDIUM = [
    r"\b(step.by.step|instructions?|guide)\s+(to|for|on\s+how\s+to)\s+(harm|hurt|injure|attack)\b",
    r"\b(personal|private|confidential)\s+(information|data|details)\s+(of|about|on)\b",
]

# ─── Hallucination signal patterns ────────────────────────────────────────────
# The LLM sometimes fabricates tool results instead of actually calling the tool.
HALLUCINATION_SIGNALS = [
    r"\[Tool result for \w+\]:",          # model inventing tool results in its own output
    r"(I ran|I executed|I searched).{0,40}(and found|which returned|result was)",
    r"https?://[^\s]+\.(fake|example\.com|test\.org)",  # obvious fake URLs
]

# ─── Refusal bypass signals ───────────────────────────────────────────────────
# Phrases that suggest the model was manipulated into bypassing its guidelines
BYPASS_SIGNALS = [
    r"(as\s+(DAN|an?\s+AI\s+without\s+restrictions)|in\s+DAN\s+mode)",
    r"(sure,?\s+here'?s?\s+how\s+to|of\s+course,?\s+I\s+can\s+(help|show)\s+you\s+(how\s+to)?)\s+(hack|exploit|harm|create\s+weapons)",
]


class OutputGuard:
    """
    Validates LLM output before it reaches the user.

    Usage:
        guard = OutputGuard()
        guard.check(response_text)  # raises OutputGuardrailError if bad
    """

    def __init__(
        self,
        min_length: int  = 1,
        max_length: int  = 10_000,
        check_toxic: bool      = True,
        check_hallucination: bool = True,
        check_pii_leakage: bool   = True,
    ):
        self.min_length          = min_length
        self.max_length          = max_length
        self.check_toxic         = check_toxic
        self.check_hallucination = check_hallucination
        self.check_pii_leakage   = check_pii_leakage

        # Compile patterns at init
        self._toxic_high   = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in TOXIC_HIGH]
        self._toxic_medium = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in TOXIC_MEDIUM]
        self._hallucination= [re.compile(p, re.IGNORECASE | re.DOTALL) for p in HALLUCINATION_SIGNALS]
        self._bypass       = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in BYPASS_SIGNALS]

        # Reuse same PII patterns as input guard
        from agent.guardrails.input_guard import PII_PATTERNS
        self._pii_re = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in PII_PATTERNS.items()
        }

    def check(self, text: str) -> None:
        """
        Run all output checks. Raises OutputGuardrailError on violation.
        Returns None if clean.
        """
        self._check_length(text)
        if self.check_toxic:
            self._check_toxicity(text)
            self._check_bypass(text)
        if self.check_hallucination:
            self._check_hallucination(text)
        if self.check_pii_leakage:
            self._check_pii_leakage(text)

        logger.info(f"[OutputGuard] ✓ passed ({len(text)} chars)")

    # ── Private checks ─────────────────────────────────────────────────────────

    def _check_length(self, text: str) -> None:
        if len(text) < self.min_length:
            raise OutputGuardrailError(
                "The model returned an empty response.",
                reason="empty_response",
                severity="low",
            )
        if len(text) > self.max_length:
            raise OutputGuardrailError(
                f"Response too long ({len(text)} chars). Possible runaway generation.",
                reason="response_too_long",
                severity="medium",
            )

    def _check_toxicity(self, text: str) -> None:
        for pattern in self._toxic_high:
            if pattern.search(text):
                logger.warning(f"[OutputGuard] ⚠ HIGH toxicity detected")
                raise OutputGuardrailError(
                    "I can't provide that response as it violates safety guidelines.",
                    reason="toxic_content_high",
                    severity="critical",
                )
        for pattern in self._toxic_medium:
            if pattern.search(text):
                logger.warning(f"[OutputGuard] ⚠ MEDIUM toxicity detected")
                raise OutputGuardrailError(
                    "My response was flagged by safety filters. Please rephrase your request.",
                    reason="toxic_content_medium",
                    severity="high",
                )

    def _check_bypass(self, text: str) -> None:
        for pattern in self._bypass:
            if pattern.search(text):
                logger.warning(f"[OutputGuard] ⚠ Refusal bypass detected in output")
                raise OutputGuardrailError(
                    "I detected that my response may have been manipulated. Blocking for safety.",
                    reason="refusal_bypass",
                    severity="critical",
                )

    def _check_hallucination(self, text: str) -> None:
        for pattern in self._hallucination:
            match = pattern.search(text)
            if match:
                logger.warning(f"[OutputGuard] ⚠ Hallucination signal: '{match.group()}'")
                raise OutputGuardrailError(
                    "I detected a possible hallucinated tool result in my response.",
                    reason="hallucination_signal",
                    severity="medium",
                )

    def _check_pii_leakage(self, text: str) -> None:
        found = []
        for pii_type, pattern in self._pii_re.items():
            # Skip email — agents may legitimately discuss emails
            if pii_type == "email":
                continue
            if pattern.search(text):
                found.append(pii_type)

        if found:
            logger.warning(f"[OutputGuard] ⚠ PII leakage in output: {found}")
            raise OutputGuardrailError(
                f"Response was blocked — it contained sensitive data ({', '.join(found)}).",
                reason="pii_leakage",
                severity="high",
            )