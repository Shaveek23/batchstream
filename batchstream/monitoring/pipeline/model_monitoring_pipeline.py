from typing import List, Dict, Tuple
import numpy as np
from ..base.model_monitoring import ModelMonitoring
from steps.base.monitoring_step import MonitoringStep



class ModelMonitoringPipeline(ModelMonitoring):
    
    def __init__(self, test_steps: List[Tuple[str, MonitoringStep]], detect_condition: str='any'):
        self.test_steps = test_steps
        self.detect_condition = detect_condition

    def monitor(self, x_history: List, y_history: List[int], prediction_history: List[int], drift_history: List[List[int]]) -> bool:
        monitoring_results: Dict[str, bool] = []
        for test_name, test in self.test_steps:
            monitoring_results.update({test_name: test.monitor(x_history, y_history, prediction_history, drift_history)})
        return self._make_is_drift_decision(list(monitoring_results.values()))

    def _make_is_drift_decision(self, monitoring_results: List[bool]):
        if self.detect_condition == 'any':
            return np.array(monitoring_results).any()
        if self.detect_condition == 'all':
            return np.array(monitoring_results).all()
