from typing import List
from ..base.stream_pipeline import StreamPipeline
from ...detectors.base.detector import DriftDetector
from ...history.base.history_manager import HistoryManager
from ...estimators.base.batch_model_estimator import BatchModelEstimator
from ...model_comparers.base.model_comparer import ModelComparer



class BatchPipeline(StreamPipeline):

    def __init__(self,
            batch_model: BatchModelEstimator,
            input_drift_detector: DriftDetector | List[DriftDetector],
            output_drift_detector: DriftDetector | List[DriftDetector],
            history: HistoryManager,
            model_comparer: ModelComparer = None,
            min_samples_retrain: int = 32,
            min_samples_first_fit: int = 32
        ) -> None:
        self._estimator = batch_model
        self._comparer = model_comparer
        self._history = history
        self._input_drift_detectors: List[DriftDetector] = input_drift_detector if isinstance(input_drift_detector, list) else [input_drift_detector]
        self._output_drift_detectors: List[DriftDetector] = output_drift_detector if isinstance(input_drift_detector, list) else [output_drift_detector]
        self._min_samples_retrain = min_samples_retrain
        self._min_samples_first_fit = min_samples_first_fit

    def handle(self, x, y: int) -> int:
        self._test_first_fit()
        self._history.update_history_x(x)
        self._handle_drift_detectors(type='in')
        prediction = self._estimator.predict(x)[0] 
        self._history.update_history_y(y)
        self._history.update_predictions(prediction)
        self._select_better_model_online(x, y, prediction)
        self._handle_drift_detectors(type='out')
        return prediction
    
    def _test_first_fit(self):
        if self._history.counter == self._min_samples_first_fit:
            self._estimator.first_fit(self._history.x_history, self._history.y_history)

    def _handle_drift_detectors(self, detector_type: str='out') -> bool:
        if self._history.counter - self._history._last_retraining < self._min_samples_retrain:
            return 
        detectors = self._output_drift_detectors if detector_type == 'out' else self._input_drift_detectors
        for i in len(detectors):
           d: DriftDetector = detectors[i]
           is_drift_detected = d.detect(self._history)
           if is_drift_detected:
               retrained_model, X_retest, y_retest = d.react(self._estimator, self._history)
               self._start_models_comparison(retrained_model, X_retest, y_retest, i, detector_type)

    def _start_models_comparison(self, retrained_model, X_retest, y_retest, detector_idx: int, detector_type: str) -> None:
        self._select_better_model_offline(retrained_model, X_retest, y_retest)
        self._comparer.trigger_online_comparing(retrained_model, self._estimator.batch_model, self._history.counter, detector_idx, detector_type)

    def _select_better_model_offline(self, retrained_model, X_retest, y_retest, detector_idx: int, detector_type:str) -> bool:
        is_replace, better_model = self._comparer.is_new_better_than_old_offline(self._estimator, retrained_model, X_retest, y_retest)
        if better_model != None and is_replace:
            self._history.update_retraining_info(self._history.counter, detector_idx, detector_type)
            self._estimator.batch_model = retrained_model

    def _select_better_model_online(self, x, y, old_model_prediction) -> bool:
        if self._comparer == None: return
        result, better_model = self._comparer.is_new_better_than_old_online(x, y, old_model_prediction)
        if result == True and better_model != None:
            self._history.update_retraining_info(self._comparer.drift_iter, self._comparer.detector_idx, self._comparer.detector_type)
            self._estimator.batch_model = better_model
        return result
   
    def get_name(self) -> str:
        return super().get_name() # TO DO
