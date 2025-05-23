# LLMProvider 인터페이스

from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def get_chain(self):
        pass
