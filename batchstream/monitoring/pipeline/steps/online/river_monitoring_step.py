from collections import Counter
from river.base import DriftDetector
from .base.monitoring_step import MonitoringStep
from history.base.history_manager import HistoryManager



class RiverMonitoringStep(MonitoringStep):

    def __init__(self, step_name: str, river_detector: DriftDetector):
        self.detector: DriftDetector = river_detector
        self.step_name: str = f'river-detector_{step_name}'

    def monitor(self, history: HistoryManager) -> bool:
        self.detector.update(history.x_history[-1]) 
        
        is_drift_detected = self.detector.drift_detected
        test_report =  self._prepare_test_output(is_drift_detected=is_drift_detected, detection_idx=history.counter)
        # TO DO:
        # log artifact - test_report
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
