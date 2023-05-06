from evidently.test_suite import TestSuite
from typing import Dict
from ..base.monitoring_step import MonitoringStep
from .....history.base.history_manager import HistoryManager
from .....batch_monitoring_strategy.base.batch_monitoring_strategy import BatchMonitoringStrategy
from .....utils.logging.base.logger_factory import LoggerFactory



class EvidentlyMonitoringStep(MonitoringStep):

    def __init__(
            self,
            evidently_test_suite_args: Dict, 
            monitoring_strategy: BatchMonitoringStrategy, 
            logger_factory: LoggerFactory,
            min_instances: int=30, 
            clock: int=32, 
            detect_condition:str='any',
            name: str=None
        ):
        self.detector_args = evidently_test_suite_args
        self.detector = TestSuite(**self.detector_args)
        self.monitoring_strategy = monitoring_strategy
        self.min_instances = min_instances
        self.clock = clock
        self.detect_condition = detect_condition
        self._name = name
        self._monitoring_logger = logger_factory.get_monitoring_logger(self._name, as_html=True)
        self._monitoring_logger.log_info(f'EvidentlyMonitoringStep - name:{self._name} - START')
       
    def monitor(self, history: HistoryManager) -> bool:
        report = None
        is_drift = False
        if history._counter > self.min_instances and history._counter % self.clock == 0:
            self._monitoring_logger.log_info(f'EvidentlyMonitoringStep - name:{self._name} - test at index: {history._counter} - START')
            ref, curr = self.monitoring_strategy.get_ref_curr(history)
            report = self._perform_test(ref, curr)
            is_drift = self._decide_concept_drift(report)
            if is_drift:
                self._monitoring_logger.log_info(f'Drift detected at: {history._counter}.')
                self._monitoring_logger.log_drift_report(self.detector, history._counter)
            self._monitoring_logger.log_info(f'EvidentlyMonitoringStep - name:{self._name} - test at index: {history._counter} - END')
        return is_drift
    
    def _perform_test(self, ref, curr):
        ref.columns = [f"{i}" for i in range(len(ref.columns))]
        curr.columns = [f"{i}" for i in range(len(curr.columns))]
        self.detector = TestSuite(**self.detector_args)
        self.detector.run(reference_data=ref, current_data=curr)
        return self.detector.as_dict()
    
    def _decide_concept_drift(self, report: dict):
        if 'ERROR' in report['summary']['by_status'] and report['summary']['by_status']['ERROR'] > 0:
            self._monitoring_logger.log_warn('ERROR status in the tests!')
        any_failed = report['summary']['failed_tests'] > 0
        all_failed = report['summary']['failed_tests'] == report['summary']['total_tests']
        if self.detect_condition == 'all':
            return all_failed
        return any_failed

    def get_params(self) -> dict:
        params = {
            'type': self.__class__.__name__,
            'min_instances': self.min_instances,
            'clock': self.clock,
            'detect_condition': self.detect_condition,
            'name': self._name
        }
        suite_tests = []
        suite_tests.extend([{t.name: t.__dict__ } for t in self.detector._inner_suite.context.tests])
        params.update({'evidently_test_suite__tests': suite_tests})
        return params
    