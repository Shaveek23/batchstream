from river import compose
from typing import List, Tuple
from ..base.stream_pipeline import StreamPipeline
from ...combine.base.pipeline_combiner import PipelineCombiner



class CombinationPipeline(StreamPipeline):
    
    def __init__(self, members: List[StreamPipeline], combiner: PipelineCombiner):
        self.members: List[StreamPipeline] = members
        self.combiner: PipelineCombiner = combiner

    def handle(self, x, y) -> Tuple[int, List[float]]:
        prediction, probabilities = self.combiner.combine(x, y, self.members)
        return prediction, probabilities

    def get_name(self):
        return "CombinationPipeline"
    
    def get_params(self) -> dict:
        return {
            'type': self.__class__.__name__,
            'members': [m.get_params() for m in self.members],
            'combiner': self.combiner.get_params()
        }
