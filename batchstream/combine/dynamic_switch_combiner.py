from river.metrics.base import Metric
from .base.pipeline_combiner import PipelineCombiner
from typing import List, Tuple
from ..pipelines.base.stream_pipeline import StreamPipeline
from ..utils.logging.base.logger_factory import LoggerFactory
from ..utils.logging.logger import Logger
import uuid



class DynamicSwitchCombiner(PipelineCombiner):

    def __init__(self, n_members: int, metric: Metric, logger_factory: LoggerFactory):
        self.name = f"{str(uuid.uuid4())[:4]}"
        self._logger: Logger = logger_factory.get_logger(f'DynamicSwitchCombiner_{self.name}')
        self._metrics = []
        self._n_members = n_members
        for i in range(n_members):
            self._metrics.append(metric.clone())

    def combine(self, x, y, members: List[StreamPipeline]) -> Tuple[int, List[float]]:
        predictions = []
        best_model_idx = self._get_the_best_model_idx()
        for i in range(len(members)):
            member = members[i]
            y_p, _ = member.handle(x, y)
            predictions.append(y_p)
            if y_p == -1 or y_p == None:
                continue
            self._metrics[i].update(y, y_p)
        self._logger.log_model_history(best_model_idx, type="switch_history")
        return predictions[best_model_idx], []

    def _get_the_best_model_idx(self):
        members_vals = [m.get() for m in self._metrics]
        max_value = max(members_vals)
        max_index = members_vals.index(max_value)
        return max_index
    
    def get_params(self):
        params = {'name': self.name, 'type': self.__class__.__name__}
        params.update({'metric_name': self._metrics[0].__class__.__name__})
        params.update({'metric': self._metrics[0].__dict__})
        params.update({'n_members': self._n_members})
        return params
