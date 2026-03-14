from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def complete(self, messages: list[dict]) -> str:
        pass