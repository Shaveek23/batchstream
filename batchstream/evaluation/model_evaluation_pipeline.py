from river.metrics.base import Metric
from typing import List, Tuple
from .base.model_evaluation import ModelEvaluation



class ModelEvaluationPipeline(ModelEvaluation):
    
    def __init__(self, metric_steps: List[Tuple[str, Metric]]):
        self.metric_steps = metric_steps

    def handle(self, y_true, y_predict):
        results = {}
        for metric_name, metric in self.metric_steps:
            metric.update(y_true, y_predict)
            metric_value = round(metric.get(), 2)
            results.update({metric_name: metric_value})
        return results
