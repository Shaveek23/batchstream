from typing import List, Tuple
import uuid
from river.utils import dict2numpy
from collections import Counter
from ..base.stream_pipeline import StreamPipeline
from ...drift_handlers.base.drift_handler import DriftHandler
from ...history.base.history_manager import HistoryManager
from ...estimators.base.batch_model_estimator import BatchModelEstimator
from ...model_comparers.base.model_comparer import ModelComparer
from ...utils.logging.base.logger_factory import LoggerFactory



class BatchPipeline(StreamPipeline):

    def __init__(self,
            batch_model: BatchModelEstimator,
            input_drift_handlers: DriftHandler | List[DriftHandler],
            output_drift_handlers: DriftHandler | List[DriftHandler],
            history: HistoryManager,
            logger_factory: LoggerFactory,
            model_comparer: ModelComparer = None,
            min_samples_retrain: int = 32,
            min_samples_first_fit: int = 32,
            initial_return: str = 'majority'
        ) -> None:
        self._counter = Counter()
        self._estimator = batch_model
        self._comparer = model_comparer
        self._history = history
        self._input_drift_handlers: List[DriftHandler] = input_drift_handlers if isinstance(input_drift_handlers, list) else [input_drift_handlers]
        self._output_drift_handlers: List[DriftHandler] = output_drift_handlers if isinstance(input_drift_handlers, list) else [output_drift_handlers]
        self._min_samples_retrain = min_samples_retrain
        self._min_samples_first_fit = min_samples_first_fit
        self.name = f"BatchPipeline_{str(uuid.uuid4())[:4]}"
        self._logger = logger_factory.get_logger(self.name)
        self._initial_return = initial_return

    def handle(self, x, y: int) -> Tuple[int, List[float]]:
        is_not_ready = self._test_first_fit()
        if is_not_ready: return self._make_result_when_not_fit(x, y)
        self._history.update_history_x(x)
        self._handle_drift_detectors(detector_type='in')
        prediction = self._estimator.predict(dict2numpy(x).reshape(1, -1))[0] 
        probabilities = self._estimator.predict_proba(dict2numpy(x).reshape(1, -1))
        self._history.update_history_y(y)
        self._history.update_predictions(prediction)
        self._select_better_model_online(x, y, prediction)
        self._handle_drift_detectors(detector_type='out')
        return prediction, probabilities
    
    def _test_first_fit(self) -> bool:
        if self._history.counter < self._min_samples_first_fit:
            return True
        if self._history.counter == self._min_samples_first_fit:
            self._logger.log_info(f'Iter={self._history.counter} First fitting of the batch model.')
            X = self._history.get_x_history_as_pd()
            Y = self._history.get_y_history_as_pd()
            self._estimator.first_fit(X, Y)
        return False
    
    def _make_result_when_not_fit(self, x, y) -> int:
        pred = -1
        if self._initial_return == 'last_label':
            pred = self._history.y_history.iloc[-1]
        if self._initial_return == 'majority':
            pred = self._majority_classifier(y)
        self._history.update_history_x(x)
        self._history.update_history_y(y)
        self._history.update_predictions(pred)
        return pred, None
    
    def _majority_classifier(self, y) -> int:
        if self._counter.total() == 0: return 0
        pred = self._counter.most_common(n=1)[0][0]
        self._counter[y] += 1
        return pred
     
    def _handle_drift_detectors(self, detector_type: str='out'):
        if self._history._last_retraining != None and self._history.counter - self._history._last_retraining < self._min_samples_retrain:
            return 
        detectors = self._output_drift_handlers if detector_type == 'out' else self._input_drift_handlers
        if detectors == None or len(detectors) < 1 or detectors[0] == None:
            return
        for i in range(len(detectors)):
           d: DriftHandler = detectors[i]
           is_drift_detected = d.detect(self._history)
           if is_drift_detected:
               self._logger.log_info(f'Iter={self._history.counter}: drift detected. {detector_type} detector num.: {i}')
               retrained_model, X_retest, y_retest = d.react(self._estimator, self._history)
               self._start_models_comparison(retrained_model, X_retest, y_retest, i, detector_type)
               return

    def _start_models_comparison(self, retrained_model, X_retest, y_retest, detector_idx: int, detector_type: str) -> None:
        if self._comparer == None:
            self._logger.log_info(f'Iter={self._history.counter}: no model selection: replacing model with the retrained one.')
            self._estimator.batch_model = retrained_model
            return
        self._logger.log_info(f'Iter={self._history.counter}: Comparing the old and the retrained model.')
        self._select_better_model_offline(retrained_model, X_retest, y_retest, detector_idx, detector_type)
        self._comparer.trigger_online_comparing(retrained_model, self._estimator.batch_model, self._history.counter, detector_idx, detector_type)

    def _select_better_model_offline(self, retrained_model, X_retest, y_retest, detector_idx: int, detector_type:str):
        is_replace, better_model = self._comparer.is_new_better_than_old_offline(retrained_model, self._estimator, X_retest, y_retest)
        if better_model != None and is_replace:
            self._logger.log_info(f'Iter={self._history.counter}: offline model selection: replacing model with the retrained one.')
            self._logger.log_model_replacement_history(self._history.counter)
            self._history.update_retraining_info(self._history.counter, detector_idx, detector_type)
            self._estimator.batch_model = retrained_model

    def _select_better_model_online(self, x, y, old_model_prediction):
        if self._comparer == None: return
        result, better_model, drift_iter, detector_idx, detector_type  = self._comparer.is_new_better_than_old_online(x, y, old_model_prediction)
        if result == True and better_model != None:
            self._logger.log_info(f'Iter={self._history.counter}: online model selection: replacing model with the retrained one.')
            self._logger.log_model_replacement_history(self._history.counter)
            self._history.update_retraining_info(drift_iter, detector_idx, detector_type)
            self._estimator.batch_model = better_model
   
    def get_name(self) -> str:
        return "BatchPipeline" # TO DO
    
    def get_params(self) -> dict:
        params = {
            'type': self.__class__.__name__,
            'min_samples_retrain': self._min_samples_retrain,
            'min_samples_first_fit': self._min_samples_first_fit,
            'initial_return': self._initial_return,
            'name': self.name
        }
        params.update({'history': self._history.get_params()})
        params.update({'batch_model': self._estimator.get_params()})
        params.update({'model_comparer': None if self._comparer == None else self._comparer.get_params()})
        params.update({'input_drift_detector': self._get_drift_detectors_params(self._input_drift_handlers)})
        params.update({'output_drift_detector': self._get_drift_detectors_params(self._output_drift_handlers)})
        return params
    
    def _get_drift_detectors_params(self, detectors) -> List:
        if len(detectors) == 1:
            if detectors[0] == None:
                return None
            return [detectors[0].get_params()]
        return [d.get_params() for d in detectors]
        