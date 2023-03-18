from abc import ABC, abstractmethod


class StreamPipeline(ABC):
    
    @abstractmethod
    def handle(self, x, y: int) -> int:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
