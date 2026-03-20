from langfuse import observe

import requests
from llm.base import BaseLLM
from config.settings import settings
from utils.retry import retry
from utils.logger import Logger

logger = Logger(__name__)

class OllamaLLM(BaseLLM):

    def __init__(self):
        self.base_url = settings.base_url
        self.model    = settings.model
        self._verify_connection()

    def _verify_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(self.model in m for m in models):
                logger.warning(f"Model '{self.model}' not found. Available: {models}")
            else:
                logger.info(f"Connected to Ollama. Using model: {self.model}")
        except Exception as e:
            logger.error(f"Cannot reach Ollama at {self.base_url}: {e}")
            raise

    @retry(max_attempts=3, delay=1.0)
    @observe()
    def complete(self, messages: list[dict]) -> str:
        # Separate system prompt — Ollama handles it differently
        chat_messages = messages

        payload = {
            "model":    self.model,
            "messages": chat_messages,
            "stream":   False,
            "options":  {"temperature": 0.7},
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=settings.timeout,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        logger.info(f"LLM responded ({len(content)} chars)")
        return content