import re
import json
from llm.base    import BaseLLM
from agent.memory import Memory
from tools.registry import registry
from config.settings import settings
from utils.logger import Logger

logger = Logger(__name__)


class Agent:
    """
    ReAct agent: Reason → Act → Observe → Repeat until done.

    Tool call format the LLM must follow:
        TOOL: tool_name({"arg1": "value1", "arg2": "value2"})

    The agent loops until:
      - LLM produces a plain response (no TOOL: call)   → done
      - max_steps is reached                             → safety exit
    """

    def __init__(self, llm: BaseLLM, system: str = ""):
        self.llm    = llm
        self.memory = Memory(system=system)

    def __call__(self, user_message: str) -> str:
        logger.info(f"User: {user_message}")
        self.memory.add("user", user_message)

        for step in range(1, settings.max_steps + 1):
            logger.info(f"--- Step {step}/{settings.max_steps} ---")
            response = self._execute()

            tool_call = self._parse_tool_call(response)

            if tool_call:
                # Agent wants to use a tool
                name   = tool_call["name"]
                kwargs = tool_call["kwargs"]
                logger.info(f"Tool call: {name}({kwargs})")

                tool_result = registry.run(name, **kwargs)
                logger.info(f"Tool result: {str(tool_result)[:200]}")

                # Add to memory so the LLM can reason over the result
                self.memory.add("assistant", response)
                self.memory.add("user", f"[Tool result for {name}]:\n{tool_result}")
            else:
                # Final answer — no tool call detected
                self.memory.add("assistant", response)
                logger.info(f"Final answer: {response[:100]}...")
                return response

        # Safety exit if max_steps exceeded
        fallback = "I reached my step limit. Please rephrase or break the task into smaller parts."
        self.memory.add("assistant", fallback)
        return fallback

    def _execute(self) -> str:
        return self.llm.complete(self.memory.get_all())

    def _parse_tool_call(self, text: str) -> dict | None:
        """
        Parse: TOOL: tool_name({"key": "value"})
        Returns {"name": str, "kwargs": dict} or None.

        Using JSON for args is much safer than regex-parsing Python syntax.
        """
        pattern = r"TOOL:\s*(\w+)\s*\((\{.*?\})\)"
        match = re.search(pattern, text, re.DOTALL)

        if not match:
            return None

        name      = match.group(1).strip()
        args_json = match.group(2).strip()

        try:
            kwargs = json.loads(args_json)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool args as JSON: {e}\nRaw: {args_json}")
            return None

        return {"name": name, "kwargs": kwargs}