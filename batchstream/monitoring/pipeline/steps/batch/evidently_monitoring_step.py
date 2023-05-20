from evidently.test_suite import TestSuite
from typing import Dict
from ..base.monitoring_step import MonitoringStep
from .....history.base.history_manager import HistoryManager
from .....batch_monitoring_strategy.base.batch_monitoring_strategy import BatchMonitoringStrategy
from .....utils.logging.base.logger_factory import LoggerFactory
import uuid


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
        self.name = f"{name}_{str(uuid.uuid4())[:4]}"
        self._monitoring_logger = logger_factory.get_monitoring_logger(self.name, as_html=True)
        self._monitoring_logger.log_info(f'EvidentlyMonitoringStep - name:{self.name} - START')
       
    def monitor(self, history: HistoryManager) -> bool:
        report = None
        is_drift = False
        if history._counter > self.min_instances and history._counter % self.clock == 0:
            ref, curr = self.monitoring_strategy.get_ref_curr(history)
            report = self._perform_test(ref, curr)
            is_drift = self._decide_concept_drift(report)
            if is_drift:
                self._monitoring_logger.log_info(f'Drift detected at: {history._counter}.')
                self._monitoring_logger.log_drift_report(self.detector, history._counter)
        return is_drift
    
    def _perform_test(self, ref, curr):
        if 'target' not in ref.columns and 'prediction' not in ref.columns:
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
            'name': self.name,
            'monitoring_strategy': self.monitoring_strategy.get_params()
        }
        params.update({'evidently_test_suite__tests': self._get_test_suite_params(self.detector)})
        return params
    
    def _get_test_suite_params(self, test_suite: TestSuite) -> dict:
        tests_metadata = []
        for test in test_suite._inner_suite.context.tests:
            test_metadata = test.metric.__dict__
            test_metadata.update({'test_name': test.get_id()})
            test_metadata.pop('context')
            tests_metadata.append(test_metadata)  
        for preset in test_suite._test_presets:
            preset_metadata = preset.__dict__
            preset_metadata.update({'preset_name': str(preset.__class__).split('.')[-1]})
            tests_metadata.append(preset_metadata) 
        return tests_metadata
    