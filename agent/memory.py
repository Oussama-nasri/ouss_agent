from config.settings import settings


class Memory:
    """
    Manages conversation history with a sliding window.

    Why sliding window?
      LLMs have a fixed context window. If you keep ALL history,
      you'll eventually overflow it and get errors. The sliding
      window silently drops oldest messages when the limit is hit,
      keeping the agent running indefinitely.
    """

    def __init__(self, system: str = ""):
        self._messages: list[dict] = []
        if system:
            self._messages.append({"role": "system", "content": system})

    def add(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})

        # Sliding window: keep system prompt + last N messages
        non_system = [m for m in self._messages if m["role"] != "system"]
        system     = [m for m in self._messages if m["role"] == "system"]

        if len(non_system) > settings.max_history:
            non_system = non_system[-settings.max_history:]
            self._messages = system + non_system

    def get_all(self) -> list[dict]:
        return self._messages.copy()

    def clear(self, keep_system: bool = True):
        if keep_system:
            self._messages = [m for m in self._messages if m["role"] == "system"]
        else:
            self._messages = []

    def __len__(self) -> int:
        return len([m for m in self._messages if m["role"] != "system"])