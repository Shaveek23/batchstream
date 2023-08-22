from river.metrics.base import Metric
from .base.pipeline_combiner import PipelineCombiner
from typing import List, Tuple
from ..pipelines.base.stream_pipeline import StreamPipeline
from ..utils.logging.base.logger_factory import LoggerFactory
from ..utils.logging.logger import Logger
import uuid
from copy import deepcopy
import numpy as np



class WeightedVoteCombiner(PipelineCombiner):

    def __init__(self, n_members: int, metric: Metric, logger_factory: LoggerFactory):
        self.name = f"{str(uuid.uuid4())[:4]}"
        self._logger: Logger = logger_factory.get_logger(f'WeightedVoteCombiner_{self.name}')
        self._metrics = []
        self._n_members = n_members
        self.counter = 0
        for i in range(n_members):
            self._metrics.append(deepcopy(metric))

    def combine(self, x, y, members: List[StreamPipeline]) -> Tuple[int, List[float]]:
        predictions = []
        model_scores = []
        for i in range(len(members)):
            member = members[i]
            y_p, _ = member.handle(x, y)
            if y_p == -1 or y_p == None:
                print("continue")
                continue
            model_scores.append(self._metrics[i].get())
            predictions.append(y_p)
            self._metrics[i].update(y, y_p)
        self.counter += 1
        if len(predictions) == 0: return 0, []
        weights = self._get_model_weights(model_scores)
        return self._get_combined_pred(predictions, weights), []

    def _get_model_weights(self, model_scores):
        return model_scores / np.sum(model_scores)
    
    def _get_combined_pred(self, predictions, weights):
        weighted_pred = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, weights)),
            axis=0,
            arr=predictions
        )
        return weighted_pred

    def get_params(self):
        params = {'name': self.name, 'type': self.__class__.__name__}
        params.update({'metric_name': self._metrics[0].__class__.__name__})
        params.update({'metric': self._metrics[0].__dict__})
        params.update({'n_members': self._n_members})
        return params
