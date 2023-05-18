from collections import Counter
from .base.pipeline_combiner import PipelineCombiner
from typing import List, Tuple
from ..pipelines.base.stream_pipeline import StreamPipeline


class MajorityVoteCombiner(PipelineCombiner):

    def combine(self, x, y, members: List[StreamPipeline]) -> Tuple[int, List[float]]:
        predictions = []
        for member in members:
            y_p, _ = member.handle(x, y)
            if y_p != -1:
                predictions.append(y_p)
        class_counts = Counter(predictions)
        majority_class = class_counts.most_common(1)[0][0]
        return majority_class
        