from dataclasses import dataclass

@dataclass
class Settings:
    model: str = "llama2:7b"
    base_url: str = "http://localhost:11434"
    max_retries: int = 3
    timeout: int = 30
    max_history: int = 20

settings = Settings()