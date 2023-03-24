from evidently.test_suite import TestSuite
from ..base.monitoring_step import MonitoringStep
from .....history.base.history_manager import HistoryManager
from .....batch_monitoring_strategy.base.batch_monitoring_strategy import BatchMonitoringStrategy



class EvidentlyMonitoringStep(MonitoringStep):

    def __init__(self, evidently_test_suite: TestSuite, monitoring_strategy: BatchMonitoringStrategy, min_instances: int=30, clock: int=32, detect_condition:str='any'):
        self.detector = evidently_test_suite
        self.monitoring_strategy = monitoring_strategy
        self.min_instances = min_instances
        self.clock = clock
        self.detect_condition = detect_condition

    def monitor(self, history: HistoryManager) -> bool:
        self.counter += 1
        report = None
        if history.counter > self.min_instances and history.counter % self.clock == 0:
            curr, ref = self.monitoring_strategy.get_curr_ref_data(history)
            self.detector.run(reference_data=ref, current_data=curr)
            report = self.detector.as_dict()
            # TO DO:
            # log artifact - report:
            return self._decide_concept_drift(report)
        return False
    
    def _decide_concept_drift(self, report: dict):
        any_failed = report['summary']['failed_tests'] > 0
        all_failed = report['summary']['failed_tests'] == report['summary']['total_tests']
        if self.detect_condition == 'all':
            return all_failed
        return any_failed
