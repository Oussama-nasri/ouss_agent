"""
Layer 5: Audit Logger

Records every significant agent event to a structured JSON log.
This is your observability layer — the feedback loop that lets you:
  - Detect attack patterns (same user triggering guards repeatedly)
  - Debug agent failures post-mortem
  - Build analytics dashboards
  - Meet compliance requirements (GDPR, SOC2, etc.)

Log format: one JSON object per line (JSONL) — easy to stream, grep, and ingest
into any log aggregator (Datadog, Loki, Elasticsearch, etc.)

Events logged:
  - user_message          — every input received
  - llm_response          — every LLM output
  - tool_call             — every tool invocation + result
  - guardrail_triggered   — every guard violation (with layer + reason)
  - agent_response        — final response sent to user
  - session_start/end     — session boundaries
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from utils.logger import Logger

logger = Logger(__name__)


class AuditLogger:
    """
    Append-only structured audit log (JSONL format).

    Usage:
        audit = AuditLogger(log_dir="./logs")
        audit.log_input("user_id_123", "What is the capital of France?")
        audit.log_guardrail("user_id_123", layer="input", reason="prompt_injection", severity="critical")
    """

    def __init__(self, log_dir: str = "./logs", session_id: str | None = None):
        self.log_dir    = Path(log_dir)
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # One file per day — easy rotation
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"audit_{today}.jsonl"

        self._log_event("session_start", {"session_id": self.session_id})
        logger.info(f"[AuditLogger] logging to {self.log_file}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def log_input(self, user_id: str, text: str) -> None:
        self._log_event("user_message", {
            "user_id":      user_id,
            "text_length":  len(text),
            "text_preview": text[:100],
        })

    def log_output(self, text: str, step: int) -> None:
        self._log_event("llm_response", {
            "step":         step,
            "text_length":  len(text),
            "text_preview": text[:100],
            "has_tool_call": "TOOL:" in text,
        })

    def log_tool_call(self, tool_name: str, kwargs: dict, result: str, duration_ms: float) -> None:
        self._log_event("tool_call", {
            "tool":        tool_name,
            "args_keys":   list(kwargs.keys()),
            "result_preview": str(result)[:100],
            "duration_ms": round(duration_ms, 2),
            "success":     not str(result).startswith("Error"),
        })

    def log_guardrail(
        self,
        layer: str,
        reason: str,
        severity: str,
        text_preview: str = "",
        user_id: str = "unknown",
    ) -> None:
        self._log_event("guardrail_triggered", {
            "layer":        layer,
            "reason":       reason,
            "severity":     severity,
            "user_id":      user_id,
            "text_preview": text_preview[:80],
        })
        logger.warning(
            f"[AuditLogger] GUARDRAIL TRIGGERED | "
            f"layer={layer} reason={reason} severity={severity}"
        )

    def log_final_response(self, text: str, total_steps: int) -> None:
        self._log_event("agent_response", {
            "text_length":  len(text),
            "text_preview": text[:100],
            "total_steps":  total_steps,
        })

    def log_error(self, error: Exception, context: str = "") -> None:
        self._log_event("agent_error", {
            "error_type": type(error).__name__,
            "message":    str(error)[:200],
            "context":    context,
        })

    def session_end(self) -> None:
        self._log_event("session_end", {"session_id": self.session_id})

    # ── Anomaly detection ──────────────────────────────────────────────────────

    def check_anomalies(self, user_id: str, window_minutes: int = 5) -> dict:
        """
        Scan recent logs to detect suspicious patterns.
        Returns a dict with anomaly flags.

        In production: run this in a background thread or async task.
        """
        cutoff = time.time() - (window_minutes * 60)
        guardrail_hits = 0
        tool_errors = 0

        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if event.get("timestamp", 0) < cutoff:
                            continue
                        if event.get("event") == "guardrail_triggered":
                            if event.get("data", {}).get("user_id") == user_id:
                                guardrail_hits += 1
                        if event.get("event") == "tool_call":
                            if not event.get("data", {}).get("success", True):
                                tool_errors += 1
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

        anomalies = {
            "guardrail_hits_5min": guardrail_hits,
            "tool_errors_5min":    tool_errors,
            "suspicious_user":     guardrail_hits >= 3,
            "tool_degradation":    tool_errors >= 5,
        }

        if anomalies["suspicious_user"]:
            logger.warning(
                f"[AuditLogger] 🚨 ANOMALY: user '{user_id}' triggered "
                f"{guardrail_hits} guardrails in {window_minutes}min"
            )

        return anomalies

    # ── Private ────────────────────────────────────────────────────────────────

    def _log_event(self, event: str, data: dict[str, Any]) -> None:
        record = {
            "timestamp":  time.time(),
            "datetime":   datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "event":      event,
            "data":       data,
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"[AuditLogger] Failed to write log: {e}")