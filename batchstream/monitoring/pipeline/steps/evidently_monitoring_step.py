from typing import List, Dict
from evidently.test_suite import TestSuite
from .base.monitoring_step import MonitoringStep
from ....evalstrategy.base.evaluation_strategy import EvaluationStrategy



class EvidentlyMonitoringStep(MonitoringStep):

    def __init__(self, evidently_test_suite: TestSuite, evaluation_strategy: EvaluationStrategy, min_instances: int=30, clock: int=32):
        self.detector = evidently_test_suite
        self.eval_strategy = evaluation_strategy
        self.min_instances = min_instances
        self.clock = clock
        self.counter = 0

    def monitor(self, x_history: List, y_history: List[int], prediction_history: List[int], drift_history: List[int]) -> dict:
        self.counter += 1
        report = None
        if self.counter > self.min_instances and self.counter % self.clock == 0:
            curr, ref = self.eval_strategy.get_curr_ref_data(x_history, y_history, prediction_history, drift_history)
            self.detector.run(reference_data=ref, current_data=curr)
            report = self.detector.as_dict()
        return report
