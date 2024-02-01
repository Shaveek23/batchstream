from river.metrics.base import Metric
from .base.pipeline_combiner import PipelineCombiner
from typing import List, Tuple
from ..pipelines.base.stream_pipeline import StreamPipeline
from ..utils.logging.base.logger_factory import LoggerFactory
from ..utils.logging.logger import Logger
import uuid
from copy import deepcopy
import numpy as np



class DiverseVoteCombiner(PipelineCombiner):

    def __init__(self, n_members: int, K: int, th: float, clock: int, metric: Metric, logger_factory: LoggerFactory):
        self.name = f"{str(uuid.uuid4())[:4]}"
        self._logger: Logger = logger_factory.get_logger(f'DiverseVoteCombiner_{self.name}')
        self._metrics = []
        self._prediction_vector_list = [[] for _ in range(0, n_members)]
        self._n_members = n_members
        self.counter = 0
        self._K = K
        self._th = th if th != None else 0.3
        self._clock = clock
        self._new_scores = [1.0] * n_members
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
        self.counter += 1

        if self.counter % self._clock == 0:
            selected_model_indices = self._get_diverse_models(model_scores)
            self._new_scores = [model_scores[i] if i in selected_model_indices else 0.0 for i in range(len(model_scores))]
            weights = self._get_model_weights(self._new_scores)
            self._log_history(weights)
            predictions = [l[-1] for l in self._prediction_vector_list]
            self._prediction_vector_list = [[] for _ in range(0, self._n_members)]
            return self._get_combined_pred(predictions, weights), []
        
        weights = self._get_model_weights(self._new_scores)
        predictions = [l[-1] for l in self._prediction_vector_list]
        return self._get_combined_pred(predictions, weights), []
    
    def _log_history(self, weights):
        line = f'cnt: {self.counter}; weights: [' + ','.join([f"{w}" for w in weights]) + ']'
        self._logger.append_to_txt(line, 'history.txt')

    def cosine_similarity(self, vector_a, vector_b):
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        similarity = dot_product / (norm_a * norm_b)
        return similarity

    def select_maximally_different_models(self, prediction_vectors, K, best_model_idx):

        num_models = len(prediction_vectors)
        selected_indices = []

        # Normalize prediction vectors
        normalized_vectors = [vector / np.linalg.norm(vector) for vector in prediction_vectors]

        # Select the best model first
        first_index = best_model_idx
        selected_indices.append(first_index)

        # Greedy selection based on cosine similarity
        while len(selected_indices) < K:
            avg_cosine_similarities = []

            for i in range(num_models):
                if i in selected_indices:
                    avg_cosine_similarities.append(float('inf'))  # Already selected
                else:
                    avg_cosine = np.mean([self.cosine_similarity(normalized_vectors[i], normalized_vectors[j])
                                        for j in selected_indices])
                    avg_cosine_similarities.append(avg_cosine)
            next_index = np.argmin(avg_cosine_similarities)
            selected_indices.append(next_index)

        return selected_indices
        
    def _get_diverse_models(self, model_scores):
        max_val = np.max(model_scores)
        th = max_val - max_val * self._th
        good_enough_model_indices = np.where(np.array(model_scores) > th)[0]
        if len(good_enough_model_indices) == 0: good_enough_model_indices = np.array([i for i in range(0, len(model_scores))])
        best_model_indices = [i for i in range(len(self._prediction_vector_list)) if i in good_enough_model_indices]  
        best_prediction_vector_list = [np.array(self._prediction_vector_list[i]) for i in range(len(self._prediction_vector_list)) if i in good_enough_model_indices] 
        best_model_idx = np.argmax(good_enough_model_indices)
        selected_indices = self.select_maximally_different_models(best_prediction_vector_list, self._K, best_model_idx)
        selected_model_indices = [best_model_indices[j] for j in selected_indices]
        return selected_model_indices

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
