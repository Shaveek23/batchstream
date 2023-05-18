from abc import ABC, abstractmethod
from typing import List, Tuple
from ...pipelines.base.stream_pipeline import StreamPipeline



class PipelineCombiner(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def combine(self, x, y, members: List[StreamPipeline]) -> Tuple[int, List[float]]:
        pass
