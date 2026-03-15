import inspect
from typing import Callable
from utils.logger import Logger

logger = Logger(__name__)


class ToolRegistry:
    """
    Central registry for all agent tools.

    - Register tools with @registry.register("name")
    - Auto-generates prompt documentation from docstrings
    - Safely dispatches tool calls with error handling
    """

    def __init__(self):
        self._tools: dict[str, Callable] = {}

    def register(self, name: str):
        """Decorator to register a function as a tool."""
        def decorator(fn: Callable):
            self._tools[name] = fn
            logger.info(f"Tool registered: {name}")
            return fn
        return decorator

    def run(self, name: str, **kwargs) -> str:
        """Execute a tool by name with given kwargs."""
        if name not in self._tools:
            available = list(self._tools.keys())
            return f"Error: tool '{name}' not found. Available tools: {available}"
        try:
            logger.info(f"Running tool: {name}({kwargs})")
            result = self._tools[name](**kwargs)
            return str(result)
        except TypeError as e:
            return f"Error: wrong arguments for tool '{name}': {e}"
        except Exception as e:
            logger.error(f"Tool '{name}' raised an error: {e}")
            return f"Tool error: {e}"

    def prompt_docs(self) -> str:
        """
        Auto-generate tool documentation for the system prompt
        from each tool's docstring. The LLM reads this to know
        what tools exist and how to call them.
        """
        docs = []
        for name, fn in self._tools.items():
            doc = inspect.getdoc(fn) or "No description."
            # Extract first line as summary, rest as detail
            lines = doc.split("\n")
            summary = lines[0]
            args_block = "\n    ".join(lines[1:]).strip()
            entry = f"- {name}: {summary}"
            if args_block:
                entry += f"\n    {args_block}"
            docs.append(entry)
        return "\n".join(docs)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())


# ─── Global registry instance ───────────────────────────────────────────────
registry = ToolRegistry()


# ─── Register all tools ──────────────────────────────────────────────────────
from tools.web_search import web_search
from tools.file_io    import read_file, write_file, list_files
from tools.code_exec  import run_python

registry.register("web_search")(web_search)
registry.register("read_file")(read_file)
registry.register("write_file")(write_file)
registry.register("list_files")(list_files)
registry.register("run_python")(run_python)