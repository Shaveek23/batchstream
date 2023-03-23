import pandas as pd
from river import stream
from typing import List
from ..pipelines.base.stream_pipeline import StreamPipeline
from ..evaluation.base.model_evaluation import ModelEvaluation
from ..utils.logging.performance_logger import PerformanceEvalLogger
from ..utils.logging.base.logger_factory import LoggerFactory



class StreamExperiment:

    def __init__(
            self,
            stream_pipeline: StreamPipeline,
            pipeline_evaluation: ModelEvaluation,
            logger_factory: LoggerFactory
        ):
        self._stream_pipeline = stream_pipeline
        self._stream_evaluation = pipeline_evaluation
        self._perf_logger: PerformanceEvalLogger = logger_factory.get_performance_logger()
        self._results: List[dict] = []

    def run(self, df: pd.DataFrame):
        y = df.pop('target')
        X = df
        for xi, yi in stream.iter_pandas(X, y):
            y_pred = int(self._stream_pipeline.handle(xi, yi))
            eval_report = self._stream_evaluation.handle(yi, y_pred)
            self._log_batch_results(eval_report)
        self._log_last_results()
        
    def _log_batch_results(self, eval_report: dict):
        self._results.append(eval_report)
        if len(self._results) == 100:
            self._perf_logger.log_info(f'Logging a batch of {len(self._results)} reports.')
            self._perf_logger.log_eval_report(self._results)
            self._results.clear()

    def _log_last_results(self):
        if len(self._results) > 0:
            self._perf_logger.log_info(f'Logging the last batch of {len(self._results)} reports.')
            self._perf_logger.log_eval_report(self._results)
            self._results.clear()
