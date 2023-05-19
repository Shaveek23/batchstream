from river.metrics.base import Metric
from typing import List, Tuple
from .base.model_evaluation import ModelEvaluation



class RiverEvaluationPipeline(ModelEvaluation):
    
    def __init__(
        self,
        metric_steps: List[Tuple[str, Metric]]
        ):
        self.metric_steps = metric_steps
        self._results_history: List[dict] = []

    def handle(self, y_true, y_predict):
        results = {}
        for metric_name, metric in self.metric_steps:
            metric.update(y_true, y_predict)
            metric_value = round(metric.get(), 4)
            results.update({metric_name: metric_value})
        return results
    
    def get_params(self) -> dict:
        step_params = []
        for metric_name, metric in self.metric_steps:
            step_params.append({metric_name: metric.__dict__})
        return {
            'type': self.__class__.__name__,
            'metric_steps': step_params
        }
