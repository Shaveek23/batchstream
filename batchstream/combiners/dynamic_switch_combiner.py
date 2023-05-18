from river.metrics.base import Metric
from .base.pipeline_combiner import PipelineCombiner
from typing import List, Tuple
from ..pipelines.base.stream_pipeline import StreamPipeline



class DynamicSwitchCombiner(PipelineCombiner):

    def __init__(self, n_members: int, metric: Metric):
        self._metrics = []
        for i in range(n_members):
            self._metrics.append(metric.clone())

    def combine(self, x, y, members: List[StreamPipeline]) -> Tuple[int, List[float]]:
        predictions = []
        best_model_idx = self._get_the_best_model_idx()
        for i in range(len(members)):
            member = members[i]
            y_p, _ = member.handle(x, y)
            predictions.append(y_p)
            if y_p != -1:
                continue
            self._metrics.update(y, y_p)
        return predictions[best_model_idx]

    def _get_the_best_model_idx(self):
        members_vals = [m.get() for m in self._metrics]
        max_value = max(members_vals)
        max_index = members_vals.index(max_value)
        return max_index
