"""
Layer 4: Tool Guard

Runs BEFORE any tool is executed.
Goal: validate arguments and enforce confirmation for risky/irreversible actions.

Risk classification matrix:
  ┌─────────────────┬──────────────┬───────────────────────────┐
  │ Tool            │ Reversible?  │ Risk Level                │
  ├─────────────────┼──────────────┼───────────────────────────┤
  │ web_search      │ Yes          │ LOW  — read-only           │
  │ list_files      │ Yes          │ LOW  — read-only           │
  │ read_file       │ Yes          │ LOW  — read-only           │
  │ run_python      │ Partial      │ MEDIUM — sandboxed exec    │
  │ write_file      │ No (overwrites)│ HIGH — destructive write │
  └─────────────────┴──────────────┴───────────────────────────┘

Why a separate layer from code_exec.py's blocklist?
  code_exec.py blocks dangerous CODE patterns (static analysis).
  ToolGuard blocks dangerous TOOL CALLS at the orchestration level.
  They are complementary, not redundant.
"""

import re
from typing import Any
from agent.guardrails.exceptions import ToolGuardrailError
from utils.logger import Logger

logger = Logger(__name__)


# ─── Tool Schemas ─────────────────────────────────────────────────────────────
# Defines required args, types, and constraints per tool.
# Add new tools here when you register them in the registry.

TOOL_SCHEMAS: dict[str, dict] = {
    "web_search": {
        "required": ["query"],
        "types":    {"query": str, "max_results": int},
        "constraints": {
            "query":       lambda v: 1 <= len(v) <= 500,
            "max_results": lambda v: 1 <= v <= 20,
        },
    },
    "read_file": {
        "required": ["filename"],
        "types":    {"filename": str},
        "constraints": {
            "filename": lambda v: len(v) <= 255 and "/" not in v and "\\" not in v,
        },
    },
    "write_file": {
        "required": ["filename", "content"],
        "types":    {"filename": str, "content": str},
        "constraints": {
            "filename": lambda v: len(v) <= 255 and "/" not in v and "\\" not in v,
            "content":  lambda v: len(v) <= 500_000,  # 500KB max
        },
    },
    "list_files": {
        "required": [],
        "types":    {},
        "constraints": {},
    },
    "run_python": {
        "required": ["code"],
        "types":    {"code": str, "timeout": int},
        "constraints": {
            "code":    lambda v: len(v) <= 10_000,
            "timeout": lambda v: 1 <= v <= 30,
        },
    },
}

# ─── Risk levels ──────────────────────────────────────────────────────────────
TOOL_RISK_LEVELS: dict[str, str] = {
    "web_search":  "low",
    "list_files":  "low",
    "read_file":   "low",
    "run_python":  "medium",
    "write_file":  "high",
}

# ─── Code patterns that are dangerous even inside run_python ─────────────────
# These supplement code_exec.py's blocklist at the orchestration level.
DANGEROUS_CODE_PATTERNS = [
    r"subprocess",
    r"__import__",
    r"compile\s*\(",
    r"globals\s*\(\s*\)",
    r"locals\s*\(\s*\)",
    r"getattr\s*\(.+,\s*['\"]__",  # dunder attribute access
    r"socket\s*\.",                 # network access
    r"urllib",
    r"requests",
    r"http\.client",
]


class ToolGuard:
    """
    Validates tool calls before execution.

    Usage:
        guard = ToolGuard(require_confirmation_for=["write_file"])
        guard.check("write_file", {"filename": "out.txt", "content": "hello"})
    """

    def __init__(
        self,
        require_confirmation_for: list[str] | None = None,
        block_high_risk: bool = False,
    ):
        """
        Args:
            require_confirmation_for: Tool names that need user confirmation.
                                      Set to None to disable interactive prompts.
            block_high_risk:          If True, high-risk tools raise an error
                                      instead of prompting (useful in automated pipelines).
        """
        self.require_confirmation_for = require_confirmation_for or []
        self.block_high_risk          = block_high_risk

        self._dangerous_code_re = [
            re.compile(p, re.IGNORECASE) for p in DANGEROUS_CODE_PATTERNS
        ]

    def check(self, tool_name: str, kwargs: dict) -> None:
        """
        Validate a tool call. Raises ToolGuardrailError on violation.
        May prompt user for confirmation (interactive mode).
        """
        self._check_tool_known(tool_name)
        self._check_schema(tool_name, kwargs)
        self._check_code_safety(tool_name, kwargs)
        self._check_risk_level(tool_name, kwargs)

        logger.info(f"[ToolGuard] ✓ {tool_name}({list(kwargs.keys())}) passed")

    # ── Private checks ─────────────────────────────────────────────────────────

    def _check_tool_known(self, tool_name: str) -> None:
        if tool_name not in TOOL_SCHEMAS:
            raise ToolGuardrailError(
                f"Unknown tool '{tool_name}'. Not in schema registry.",
                reason="unknown_tool",
                severity="high",
            )

    def _check_schema(self, tool_name: str, kwargs: dict) -> None:
        schema = TOOL_SCHEMAS[tool_name]

        # Check required args
        for required_key in schema["required"]:
            if required_key not in kwargs:
                raise ToolGuardrailError(
                    f"Tool '{tool_name}' is missing required argument: '{required_key}'.",
                    reason="missing_required_arg",
                    severity="medium",
                )

        # Check types
        for key, expected_type in schema["types"].items():
            if key in kwargs and not isinstance(kwargs[key], expected_type):
                raise ToolGuardrailError(
                    f"Tool '{tool_name}' argument '{key}' must be {expected_type.__name__}, "
                    f"got {type(kwargs[key]).__name__}.",
                    reason="invalid_arg_type",
                    severity="medium",
                )

        # Check constraints
        for key, constraint_fn in schema["constraints"].items():
            if key in kwargs:
                try:
                    if not constraint_fn(kwargs[key]):
                        raise ToolGuardrailError(
                            f"Tool '{tool_name}' argument '{key}' failed validation "
                            f"(value: {repr(kwargs[key])[:50]}).",
                            reason="constraint_violation",
                            severity="medium",
                        )
                except ToolGuardrailError:
                    raise
                except Exception as e:
                    raise ToolGuardrailError(
                        f"Tool '{tool_name}' argument '{key}' validation error: {e}",
                        reason="constraint_error",
                        severity="medium",
                    )

    def _check_code_safety(self, tool_name: str, kwargs: dict) -> None:
        """Extra safety layer specifically for run_python."""
        if tool_name != "run_python":
            return

        code = kwargs.get("code", "")
        for pattern in self._dangerous_code_re:
            if pattern.search(code):
                raise ToolGuardrailError(
                    f"Code blocked at orchestration level: contains dangerous pattern.",
                    reason="dangerous_code_pattern",
                    severity="critical",
                )

    def _check_risk_level(self, tool_name: str, kwargs: dict) -> None:
        risk = TOOL_RISK_LEVELS.get(tool_name, "medium")

        if risk == "high":
            if self.block_high_risk:
                raise ToolGuardrailError(
                    f"Tool '{tool_name}' is classified as high-risk and is blocked "
                    "in automated mode.",
                    reason="high_risk_blocked",
                    severity="high",
                )

            if tool_name in self.require_confirmation_for:
                self._prompt_confirmation(tool_name, kwargs)

    def _prompt_confirmation(self, tool_name: str, kwargs: dict) -> None:
        """
        Ask the user to confirm before executing a high-risk tool.
        In a non-interactive (API) context, block instead.
        """
        print(f"\n⚠️  [ToolGuard] High-risk action requested:")
        print(f"   Tool: {tool_name}")
        for k, v in kwargs.items():
            preview = str(v)[:80] + ("..." if len(str(v)) > 80 else "")
            print(f"   {k}: {preview}")

        try:
            answer = input("\nAllow this action? (yes/no): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "no"

        if answer not in ("yes", "y"):
            raise ToolGuardrailError(
                f"User declined execution of '{tool_name}'.",
                reason="user_declined",
                severity="low",
            )
        logger.info(f"[ToolGuard] User confirmed '{tool_name}'")