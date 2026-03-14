from config.settings import settings

class Memory:
    def __init__(self):
        self._messages: list[dict] = []

    def add(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})
        # Sliding window to avoid context overflow
        if len(self._messages) > settings.max_history:
            self._messages = self._messages[-settings.max_history:]

    def get_all(self) -> list[dict]:
        return self._messages.copy()

    def clear(self):
        self._messages = []