from typing import Callable

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}

    def register(self, name: str):
        """Decorator to register tools"""
        def decorator(fn: Callable):
            self._tools[name] = fn
            return fn
        return decorator

    def run(self, name: str, **kwargs):
        if name not in self._tools:
            return f"Error: tool '{name}' not found"
        return self._tools[name](**kwargs)

    def descriptions(self) -> str:
        return "\n".join(f"- {name}" for name in self._tools)

registry = ToolRegistry()

# Register tools
@registry.register("calculator")
def calculator(expression: str) -> str:
    try:
        return str(eval(expression))  # Use safer eval in prod
    except Exception as e:
        return f"Error: {e}"

@registry.register("get_time")
def get_time() -> str:
    from datetime import datetime
    return datetime.now().isoformat()