from ..performance_logger import PerformanceEvalLogger



class LoggerFactory:

    def __init__(self,  experiment_id: str):
        self._experiment_id = experiment_id

    def get_performance_logger(self, module_name: str='performance_eval') -> PerformanceEvalLogger:
        return PerformanceEvalLogger(self._experiment_id, module_name)
    