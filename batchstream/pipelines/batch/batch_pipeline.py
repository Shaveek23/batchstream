from typing import List
from river.utils import dict2numpy
from ..base.stream_pipeline import StreamPipeline
from ...detectors.base.detector import DriftDetector
from ...history.base.history_manager import HistoryManager
from ...estimators.base.batch_model_estimator import BatchModelEstimator
from ...model_comparers.base.model_comparer import ModelComparer
from ...utils.logging.base.logger_factory import LoggerFactory



class BatchPipeline(StreamPipeline):

    def __init__(self,
            batch_model: BatchModelEstimator,
            input_drift_detector: DriftDetector | List[DriftDetector],
            output_drift_detector: DriftDetector | List[DriftDetector],
            history: HistoryManager,
            logger_factory: LoggerFactory,
            model_comparer: ModelComparer = None,
            min_samples_retrain: int = 32,
            min_samples_first_fit: int = 32,
            initial_return: str = 'minus_one'
        ) -> None:
        self._estimator = batch_model
        self._comparer = model_comparer
        self._history = history
        self._input_drift_detectors: List[DriftDetector] = input_drift_detector if isinstance(input_drift_detector, list) else [input_drift_detector]
        self._output_drift_detectors: List[DriftDetector] = output_drift_detector if isinstance(input_drift_detector, list) else [output_drift_detector]
        self._min_samples_retrain = min_samples_retrain
        self._min_samples_first_fit = min_samples_first_fit
        self._logger = logger_factory.get_logger('BatchPipeline')
        self._initial_return = initial_return

    def handle(self, x, y: int) -> int:
        is_not_ready = self._test_first_fit()
        if is_not_ready: return self._make_result_when_not_fit(x, y)
        self._history.update_history_x(x)
        self._handle_drift_detectors(detector_type='in')
        prediction = self._estimator.predict(dict2numpy(x).reshape(1, -1))[0] 
        self._history.update_history_y(y)
        self._history.update_predictions(prediction)
        self._select_better_model_online(x, y, prediction)
        self._handle_drift_detectors(detector_type='out')
        return prediction
    
    def _test_first_fit(self) -> bool:
        if self._history.counter < self._min_samples_first_fit:
            return True
        if self._history.counter == self._min_samples_first_fit:
            self._logger.log_info(f'Iter={self._history.counter} First fitting of the batch model.')
            self._estimator.first_fit(self._history.x_history, self._history.y_history)
        return False
    
    def _make_result_when_not_fit(self, x, y) -> int:
        pred = -1
        if self._initial_return == 'last_label':
            pred = self._history.y_history.iloc[-1]
        self._history.update_history_x(x)
        self._history.update_history_y(y)
        self._history.update_predictions(pred)
        return pred
        

    def _handle_drift_detectors(self, detector_type: str='out'):
        if self._history._last_retraining != None and self._history.counter - self._history._last_retraining < self._min_samples_retrain:
            return 
        detectors = self._output_drift_detectors if detector_type == 'out' else self._input_drift_detectors
        if detectors == None or len(detectors) < 1:
            self._logger.log_info(f'Iter={self._history.counter}: _handle_drift_detectors: no ({detector_type}) detectors.')
            return
        for i in range(len(detectors)):
           d: DriftDetector = detectors[i]
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
        self._select_better_model_offline(retrained_model, X_retest, y_retest)
        self._comparer.trigger_online_comparing(retrained_model, self._estimator.batch_model, self._history.counter, detector_idx, detector_type)

    def _select_better_model_offline(self, retrained_model, X_retest, y_retest, detector_idx: int, detector_type:str):
        is_replace, better_model = self._comparer.is_new_better_than_old_offline(self._estimator, retrained_model, X_retest, y_retest)
        if better_model != None and is_replace:
            self._logger.log_info(f'Iter={self._history.counter}: offline model selection: replacing model with the retrained one.')
            self._history.update_retraining_info(self._history.counter, detector_idx, detector_type)
            self._estimator.batch_model = retrained_model

    def _select_better_model_online(self, x, y, old_model_prediction):
        if self._comparer == None: return
        result, better_model = self._comparer.is_new_better_than_old_online(x, y, old_model_prediction)
        if result == True and better_model != None:
            self._logger.log_info(f'Iter={self._history.counter}: online model selection: replacing model with the retrained one.')
            self._history.update_retraining_info(self._comparer.drift_iter, self._comparer.detector_idx, self._comparer.detector_type)
            self._estimator.batch_model = better_model
   
    def get_name(self) -> str:
        return "BatchPipeline" # TO DO
