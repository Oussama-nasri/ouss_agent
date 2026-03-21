class GuardrailError(Exception):
    """Base class for all guardrail violations."""

    def __init__(self, message: str, layer: str, reason: str, severity: str = "medium"):
        super().__init__(message)
        self.layer    = layer     # "input" | "output" | "tool"
        self.reason   = reason    # machine-readable category
        self.severity = severity  # "low" | "medium" | "high" | "critical"

    def to_dict(self) -> dict:
        return {
            "error":    self.__class__.__name__,
            "message":  str(self),
            "layer":    self.layer,
            "reason":   self.reason,
            "severity": self.severity,
        }


class InputGuardrailError(GuardrailError):
    """Raised when user input violates policy."""
    def __init__(self, message: str, reason: str, severity: str = "high"):
        super().__init__(message, layer="input", reason=reason, severity=severity)


class OutputGuardrailError(GuardrailError):
    """Raised when LLM output violates policy."""
    def __init__(self, message: str, reason: str, severity: str = "medium"):
        super().__init__(message, layer="output", reason=reason, severity=severity)


class ToolGuardrailError(GuardrailError):
    """Raised when a tool call violates policy."""
    def __init__(self, message: str, reason: str, severity: str = "high"):
        super().__init__(message, layer="tool", reason=reason, severity=severity)