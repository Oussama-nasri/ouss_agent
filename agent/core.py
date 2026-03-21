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
    ReAct agent with layered guardrails.

    Args:
        llm:        Any BaseLLM implementation (Ollama, OpenAI, etc.)
        system:     System prompt string
        guardrails: GuardrailPipeline instance. Pass None to disable (dev mode).
        user_id:    Identifier for audit logging (session, user email, etc.)
    """

    def __init__(
            self,
            llm: BaseLLM,
            system: str = "",
            guardrails: GuardrailPipeline | None = None,
            user_id: str = "anonymous",
    ):
        self.llm = llm
        self.memory = Memory(system=system)
        self.guardrails = guardrails
        self.user_id = user_id

    def __call__(self, user_message: str) -> str:
        # ── Layer 1: Input guard ───────────────────────────────────────────────
        if self.guardrails:
            try:
                self.guardrails.check_input(user_message, user_id=self.user_id)
            except GuardrailError as e:
                logger.warning(f"Input blocked: {e.reason}")
                return str(e)

        logger.info(f"User: {user_message}")
        self.memory.add("user", user_message)

        total_steps = 0

        for step in range(1, settings.max_steps + 1):
            total_steps = step
            logger.info(f"--- Step {step}/{settings.max_steps} ---")

            response = self._execute()

            # ── Layer 3: Output guard ──────────────────────────────────────────
            if self.guardrails:
                try:
                    self.guardrails.check_output(response, step=step)
                except GuardrailError as e:
                    logger.warning(f"Output blocked at step {step}: {e.reason}")
                    safe_msg = "I wasn't able to generate a safe response. Please rephrase your request."
                    self.memory.add("assistant", safe_msg)
                    return safe_msg

            tool_call = self._parse_tool_call(response)

            if tool_call:
                name = tool_call["name"]
                kwargs = tool_call["kwargs"]

                # ── Layer 4: Tool guard ────────────────────────────────────────
                if self.guardrails:
                    try:
                        self.guardrails.check_tool(name, kwargs)
                    except GuardrailError as e:
                        logger.warning(f"Tool blocked: {name} — {e.reason}")
                        block_msg = f"[Tool '{name}' was blocked: {e}]"
                        self.memory.add("assistant", response)
                        self.memory.add("user", block_msg)
                        continue

                # ── Execute tool ───────────────────────────────────────────────
                t0 = time.monotonic()
                tool_result = registry.run(name, **kwargs)
                duration_ms = (time.monotonic() - t0) * 1000

                if self.guardrails:
                    self.guardrails.log_tool_result(name, kwargs, tool_result, duration_ms)

                logger.info(f"Tool '{name}' → {str(tool_result)[:150]} ({duration_ms:.0f}ms)")

                self.memory.add("assistant", response)
                self.memory.add("user", f"[Tool result for {name}]:\n{tool_result}")

            else:
                # ── Final answer ───────────────────────────────────────────────
                self.memory.add("assistant", response)

                if self.guardrails:
                    self.guardrails.log_final(response, total_steps)

                logger.info(f"Final answer ({len(response)} chars, {total_steps} steps)")
                return response

        # ── Max steps exceeded ─────────────────────────────────────────────────
        fallback = "I reached my step limit. Please rephrase or break the task into smaller parts."
        self.memory.add("assistant", fallback)
        if self.guardrails:
            self.guardrails.log_final(fallback, total_steps)
        return fallback

    def _execute(self) -> str:
        return self.llm.complete(self.memory.get_all())

    def _parse_tool_call(self, text: str) -> dict | None:
        pattern = r"TOOL:\s*(\w+)\s*\((\{.*?\})\)"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None
        name = match.group(1).strip()
        args_json = match.group(2).strip()
        try:
            kwargs = json.loads(args_json)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool args: {e}")
            return None
        return {"name": name, "kwargs": kwargs}