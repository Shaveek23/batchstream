from .logger import Logger
from evidently.test_suite import TestSuite
import os



class MonitoringLogger(Logger):

    def __init__(self, experiment_id: str, module: str, log_dir_path: str=None, out_dir_path: str=None, as_html=False):
        super().__init__(experiment_id, module, log_dir_path, out_dir_path)
        self._as_html = as_html

    def log_drift_report(self, suite: TestSuite, counter: int):
        file_dir = os.path.join(self._output_dir, self._module)
        file_path = os.path.join(file_dir, str(counter))
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        suite.save_json(f'{file_path}.json')
        if self._as_html:
            suite.save_html(f'{file_path}.html')