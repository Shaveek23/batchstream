from river.metrics.base import Metric
from .base.pipeline_combiner import PipelineCombiner
from typing import List, Tuple
from ..pipelines.base.stream_pipeline import StreamPipeline
from ..pipelines.batch.batch_pipeline import BatchPipeline
from ..utils.logging.base.logger_factory import LoggerFactory
from ..utils.logging.logger import Logger
import uuid
import numpy as np
from copy import deepcopy


class SimilarityGroupingCombiner(PipelineCombiner):

    def __init__(self, n_members: int, n_wait: int, similarity_threshold: float, similarity_penalty: float, metric: Metric, logger_factory: LoggerFactory):
        self.name = f"{str(uuid.uuid4())[:4]}"
        self._logger: Logger = logger_factory.get_logger(f'SimilarityGroupingCombiner_{self.name}')
        self._metrics = []
        self._prediction_vector_list = [[] for _ in range(0, n_members)]
        self.counter = 0
        self._replacement_detected = False
        self._n_wait = n_wait
        self._similarity_threshold = similarity_threshold
        self._similarity_penalty = similarity_penalty
        self._from_replacement_clock = 0
        self._n_members = n_members
        self._weights = [1 / n_members for _ in range(0, n_members)]
        self._original_empty_metric = metric
        for i in range(n_members):
            self._metrics.append(deepcopy(metric))

    def combine(self, x, y, members: List[StreamPipeline]) -> Tuple[int, List[float]]:
        model_scores = []
        for i in range(len(members)):
            member = members[i]
            y_p, _ = member.handle(x, y)
            if y_p == -1 or y_p == None:
                y_p = 0
            model_scores.append(self._metrics[i].get())
            self._prediction_vector_list[i].append(y_p)
            self._metrics[i].update(y, y_p)
        predictions = [l[-1] for l in self._prediction_vector_list]
        self._adapt_to_possible_changes(members, model_scores)  
        self.counter += 1
        return self._get_combined_pred(predictions, self._weights), []
    
    def _adapt_to_possible_changes(self, members: List[StreamPipeline], model_scores: List[float]):
        if self._replacement_detected == True:
            self._from_replacement_clock += 1
            if self._from_replacement_clock % self._n_wait == 0:
                self._replacement_detected = False
                self._from_replacement_clock = 0
                self._weights = self._get_group_weights(self._prediction_vector_list, model_scores, self._similarity_threshold, self._similarity_penalty)
                self._prediction_vector_list = [[] for _ in range(0, self._n_members)]
                metrics_temp = []
                for i in range(self._n_members):
                    metrics_temp.append(deepcopy(self._original_empty_metric))
                self._metrics = metrics_temp
        elif self._detect_any_member_change(members):
            self._replacement_detected = True

    def _detect_any_member_change(self, members: List[StreamPipeline]) -> bool:
        for member in members:
            if isinstance(member, BatchPipeline):
                if self.counter == member._history.get_last_replacement_idx():
                    return True
        return False

    def _calculate_difference_matrix(self, predictions):
        num_models = len(predictions)
        difference_matrix = [[0] * num_models for _ in range(num_models)]
        prediction_count = len(predictions[0])
        for i in range(num_models):
            for j in range(i + 1, num_models):
                num_differences = sum(1 for x, y in zip(predictions[i], predictions[j]) if x != y)
                difference_matrix[i][j] = num_differences / prediction_count
                difference_matrix[j][i] = num_differences / prediction_count
        return difference_matrix

    def _normalize_matrix(self, matrix):
        total_sum = np.sum(matrix)
        normalized_matrix = matrix / total_sum
        return normalized_matrix

    def _get_group_weights(self, predictions, metrics, similarity_threshold, similarity_penalty):
        num_models = len(predictions)
        D = self._calculate_difference_matrix(predictions)
        grouped_models = set()
        weights = np.zeros(num_models)
        metrics_copy = deepcopy(metrics)
        while len(grouped_models) < num_models:
            best_model_idx = np.argmax(metrics_copy)
            weights[best_model_idx] = metrics_copy[best_model_idx]
            metrics_copy[best_model_idx] = -np.inf
            grouped_models.add(best_model_idx)
            best_model_differences = D[best_model_idx]
            for idx, difference in enumerate(best_model_differences):
                if difference <= similarity_threshold and idx not in grouped_models:
                    grouped_models.add(idx)
                    model_weight = (1.0 - (1.0 - difference) * similarity_penalty) * metrics_copy[idx]
                    model_weight = model_weight if model_weight > 0 else 0.0
                    weights[idx] = model_weight
                    metrics_copy[idx] = -np.inf
        weights = self._normalize_matrix(weights)
        self.log_change_in_history(weights, D, metrics)
        return weights
    
    def log_change_in_history(self, weights, D, metrics):
        line = f'cnt: {self.counter};weights: [' + ','.join([f'{w}' for w in weights]) + \
            '];diffs: [' + ','.join([f'{d}' for d in D]) + '];scores: [' + \
                ','.join([f'{s}' for s in metrics]) + ']'
        self._logger.append_to_txt(line, 'history.txt')

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
        params.update({'n_wait': self._n_wait})
        params.update({'similarity_threshold': self._similarity_threshold})
        params.update({'similarity_penalty': self._similarity_penalty})
        return params
