from typing import List, Dict, Tuple
import numpy as np
from ..base.drift_monitoring import DriftMonitoring
from .steps.base.monitoring_step import MonitoringStep
from ...history.base.history_manager import HistoryManager



class DriftMonitoringPipeline(DriftMonitoring):
    
    def __init__(self, test_steps: List[Tuple[str, MonitoringStep]], detect_condition: str='any'):
        self.test_steps = test_steps
        self.detect_condition = detect_condition

    def monitor(self, history: HistoryManager) -> bool:
        monitoring_results: Dict[str, dict] = {}
        for test_name, test in self.test_steps:
            monitoring_results.update({test_name: test.monitor(history)})
        return self._make_is_drift_decision(list(monitoring_results.values()))

    def _make_is_drift_decision(self, monitoring_results: List[bool]):
        if self.detect_condition == 'any':
            return np.array(monitoring_results).any()
        if self.detect_condition == 'all':
            return np.array(monitoring_results).all()
        
    def get_name(self):
        return "ModelMonitoringPipeline"
    
    def get_params(self) -> dict:
        params = []
        for t in self.test_steps:
            params.append({t[0]: t[1].get_params()})
        return {
            'type': self.__class__.__name__,
            'detect_condition': self.detect_condition,
            'test_steps': params
        }
