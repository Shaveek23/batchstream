from abc import ABC, abstractmethod
from typing import List, Tuple



class StreamPipeline(ABC):
    
    @abstractmethod
    def handle(self, x, y: int) -> Tuple[int, List[float]]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass
    