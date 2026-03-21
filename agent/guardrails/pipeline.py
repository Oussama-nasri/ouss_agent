"""
Guardrail Pipeline

Single entry point that wires all layers together.
The agent only talks to this — never to individual guards directly.

This is the "façade" pattern: hide complexity behind a clean interface.
The agent shouldn't need to know which guard caught what — it just calls
pipeline.check_input() and pipeline.check_output() and handles GuardrailError.
"""

import time
from agent.guardrails.input_guard   import InputGuard
from agent.guardrails.output_guard  import OutputGuard
from agent.guardrails.tool_guard    import ToolGuard
from agent.guardrails.audit_logger  import AuditLogger
from agent.guardrails.exceptions    import GuardrailError
from utils.logger import Logger

logger = Logger(__name__)


class GuardrailPipeline:
    """
    Unified guardrail interface for the agent.

    Usage:
        pipeline = GuardrailPipeline()
        pipeline.check_input(user_text, user_id="user_123")
        pipeline.check_output(llm_response, step=1)
        pipeline.check_tool("write_file", {"filename": "x.txt", "content": "..."})
        pipeline.log_tool_result("write_file", kwargs, result, duration_ms)
        pipeline.log_final(response, total_steps)
    """

    def __init__(
        self,
        # Input guard config
        input_max_length: int  = 4000,
        check_pii_input: bool  = True,
        pii_action: str        = "warn",     # "warn" or "block"

        # Output guard config
        check_toxic: bool      = True,
        check_hallucination: bool = True,

        # Tool guard config
        require_confirmation_for: list[str] | None = None,
        block_high_risk: bool  = False,      # True for automated/CI pipelines

        # Audit config
        log_dir: str           = "./logs",
        session_id: str | None = None,
    ):
        self.input_guard  = InputGuard(
            max_length=input_max_length,
            check_pii=check_pii_input,
            pii_action=pii_action,
        )
        self.output_guard = OutputGuard(
            check_toxic=check_toxic,
            check_hallucination=check_hallucination,
        )
        self.tool_guard   = ToolGuard(
            require_confirmation_for=require_confirmation_for,
            block_high_risk=block_high_risk,
        )
        self.audit        = AuditLogger(log_dir=log_dir, session_id=session_id)

        logger.info("[GuardrailPipeline] initialized — all 5 layers active")

    # ── Layer 1: Input ─────────────────────────────────────────────────────────

    def check_input(self, text: str, user_id: str = "unknown") -> None:
        """
        Run input guard. Raises GuardrailError if blocked.
        Always logs the attempt (even clean inputs).
        """
        self.audit.log_input(user_id, text)
        try:
            self.input_guard.check(text)
        except GuardrailError as e:
            self.audit.log_guardrail(
                layer=e.layer,
                reason=e.reason,
                severity=e.severity,
                text_preview=text[:80],
                user_id=user_id,
            )
            raise

    # ── Layer 3: Output ────────────────────────────────────────────────────────

    def check_output(self, text: str, step: int = 0) -> None:
        """
        Run output guard. Raises GuardrailError if blocked.
        """
        self.audit.log_output(text, step=step)
        try:
            self.output_guard.check(text)
        except GuardrailError as e:
            self.audit.log_guardrail(
                layer=e.layer,
                reason=e.reason,
                severity=e.severity,
                text_preview=text[:80],
            )
            raise

    # ── Layer 4: Tool ──────────────────────────────────────────────────────────

    def check_tool(self, tool_name: str, kwargs: dict) -> None:
        """
        Run tool guard before execution. Raises GuardrailError if blocked.
        """
        try:
            self.tool_guard.check(tool_name, kwargs)
        except GuardrailError as e:
            self.audit.log_guardrail(
                layer=e.layer,
                reason=e.reason,
                severity=e.severity,
                text_preview=f"{tool_name}({list(kwargs.keys())})",
            )
            raise

    def log_tool_result(
        self,
        tool_name: str,
        kwargs: dict,
        result: str,
        duration_ms: float,
    ) -> None:
        """Log a completed tool call to the audit trail."""
        self.audit.log_tool_call(tool_name, kwargs, result, duration_ms)

    # ── Layer 5: Final response ────────────────────────────────────────────────

    def log_final(self, response: str, total_steps: int) -> None:
        self.audit.log_final_response(response, total_steps)

    def log_error(self, error: Exception, context: str = "") -> None:
        self.audit.log_error(error, context)

    # ── Anomaly detection ──────────────────────────────────────────────────────

    def check_anomalies(self, user_id: str) -> dict:
        return self.audit.check_anomalies(user_id)

    def end_session(self) -> None:
        self.audit.session_end()