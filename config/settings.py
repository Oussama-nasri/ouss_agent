import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama2:7b"))
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    max_retries: int = 3
    timeout: int = field(default_factory=lambda: int(os.getenv("TIMEOUT", 60)))
    max_history: int = field(default_factory=lambda: int(os.getenv("MAX_HISTORY", 20)))
    max_steps: int = field(default_factory=lambda: int(os.getenv("MAX_STEPS", 7)))


settings = Settings()