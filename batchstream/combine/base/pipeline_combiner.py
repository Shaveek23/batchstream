from abc import ABC, abstractmethod
from typing import List, Tuple
from ...pipelines.base.stream_pipeline import StreamPipeline



class PipelineCombiner(ABC):

    @abstractmethod
    def combine(self, x, y, members: List[StreamPipeline]) -> Tuple[int, List[float]]:
        pass

    @abstractmethod
    def get_params(self):
        pass
