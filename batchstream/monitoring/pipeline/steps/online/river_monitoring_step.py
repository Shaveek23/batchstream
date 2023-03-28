from collections import Counter
from river.base import DriftDetector
from batchstream.utils.logging.base.logger_factory import LoggerFactory
from ..base.monitoring_step import MonitoringStep
from history.base.history_manager import HistoryManager



class RiverMonitoringStep(MonitoringStep):

    def __init__(self, step_name: str, river_detector: DriftDetector, logger_factory: LoggerFactory):
        self.detector: DriftDetector = river_detector
        self.step_name: str = f'river-detector_{step_name}'
        self._logger = logger_factory.get_monitoring_logger(self.step_name, as_html=False)

    def monitor(self, history: HistoryManager) -> bool:
        self.detector.update(history.x_history[-1]) 
        is_drift_detected = self.detector.drift_detected
        test_report = self._prepare_test_output(is_drift_detected=is_drift_detected, detection_idx=history._counter)
        if is_drift_detected:
                self._monitoring_logger.log_info(f'Drift detected at: {history._counter}.')
                self._monitoring_logger.log_drift_report(self.detector, history._counter)
        return is_drift_detected

    def _prepare_test_output(self, is_drift_detected: bool, detection_idx: int):
        if not is_drift_detected:
            return None
        return {
            'tests': [{
                'name': self.step_name,
                'description': f'Drift detected at idx.: {detection_idx}',
                'status': 'FAIL',
                'group': 'river-detector',
                'parameters': {
                    'detected_at_idx': detection_idx,
                    'detector_type': type(self.detector)
                }.update({item for item in vars(self.detector).items() if not item[0].startswith('_')})
            }],
            'summary': {
                'all_passed': 0,
                'total_tests': 1,
                'success_tests': 0,
                'failed_tests': 1,
                'by_status': Counter({'SUCCESS': 0, 'FAIL': 1})}
        }

    def get_params(self) -> dict:
        d = self.detector._get_params()
        return {
            'type': self.__class__.__name__,
            'step_name': self.step_name,
            'river_detector': {
                'type': self.detector.__class__.__name__
            }.update(d)
        }