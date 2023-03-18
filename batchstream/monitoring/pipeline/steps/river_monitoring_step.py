from collections import Counter
from river.base import DriftDetector
from .base.monitoring_step import MonitoringStep
from typing import List, Dict



class RiverMonitoringStep(MonitoringStep):

    def __init__(self, step_name: str, river_detector: DriftDetector):
        self.detector: DriftDetector = river_detector
        self.step_name: str = f'river-detector_{step_name}'
        self.counter: int = 0

    def monitor(self, x_history: List, y_history: List[int]=None, prediction_history: List[int]=None, drift_history: List[int]=None) -> dict:
        self.counter += 1
        self.detector.update(x_history[-1])
        is_drift_detected = self.detector.drift_detected
        return self._prepare_test_output(is_drift_detected=is_drift_detected, detection_idx=self.counter)

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
