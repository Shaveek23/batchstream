from abc import ABC, abstractmethod
from typing import List
from batchstream.pipelines.base.stream_pipeline import StreamPipeline



class PipelineCombiner(ABC):

    @abstractmethod
    def handle(self, x, y, members: List[StreamPipeline]) -> int:
        pass
