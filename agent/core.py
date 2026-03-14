import re
from utils.logger import Logger
from llm.base import BaseLLM
from agent.memory import Memory
from agent.tools import registry

logger = Logger(__name__)

class Agent:
    def __init__(self, llm: BaseLLM, system: str = ""):
        self.llm = llm
        self.memory = Memory()
        self.system = system

        if system:
            self.memory.add("system", system)

    def __call__(self, user_message: str) -> str:
        logger.info(f"User: {user_message}")
        self.memory.add("user", user_message)

        # ReAct loop: Reason → Act → Observe
        for step in range(5):  # max steps guard
            response = self._execute()

            # Check if agent wants to use a tool
            tool_call = self._parse_tool_call(response)
            if tool_call:
                tool_result = registry.run(**tool_call)
                logger.info(f"Tool {tool_call['name']} → {tool_result}")
                self.memory.add("assistant", response)
                self.memory.add("user", f"Tool result: {tool_result}")
            else:
                # Final answer
                self.memory.add("assistant", response)
                logger.info(f"Agent: {response}")
                return response

        return "Max steps reached."

    def _execute(self) -> str:
        return self.llm.complete(self.memory.get_all())

    def _parse_tool_call(self, text: str) -> dict | None:
        """Simple pattern: TOOL: calculator(expression='2+2')"""
        match = re.search(r"TOOL: (\w+)\((.*)?\)", text)
        if match:
            name = match.group(1)
            args_str = match.group(2)
            # Parse simple key=value args
            args = {}
            for pair in re.findall(r"(\w+)='([^']*)'", args_str):
                args[pair[0]] = pair[1]
            return {"name": name, **args}
        return None