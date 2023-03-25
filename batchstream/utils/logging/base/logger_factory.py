from ..performance_logger import PerformanceEvalLogger
from ..monitoring_logger import MonitoringLogger
from ..logger import Logger


class LoggerFactory:

    def __init__(self,  experiment_id: str):
        self._experiment_id = experiment_id

    def get_performance_logger(self, module_name: str='performance_eval') -> PerformanceEvalLogger:
        return PerformanceEvalLogger(self._experiment_id, module_name)
    
    def get_monitoring_logger(self, module_name: str, as_html: bool=False) -> MonitoringLogger:
        return MonitoringLogger(self._experiment_id, module_name, as_html=as_html)
    
    def get_logger(self, module_name: str) -> Logger:
        return Logger(self._experiment_id, module_name)