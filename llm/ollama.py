import requests
from .base import BaseLLM
from config.settings import settings
from langfuse import observe

class OllamaLLM(BaseLLM):
    @observe()
    def complete(self, messages: list[dict]) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        chat_messages = [m for m in messages if m["role"] != "system"]

        response = requests.post(
            f"{settings.base_url}/api/chat",
            json={
                "model": settings.model,
                "messages": chat_messages,
                "system": system,
                "stream": False,
            },
            timeout=settings.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]