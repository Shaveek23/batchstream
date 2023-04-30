import pandas as pd
from river import stream
from typing import List
import time
from tqdm import tqdm
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
        self._logger = logger_factory.get_logger('experiment_metadata')
        self._results: List[dict] = []

    def run(self, df: pd.DataFrame):
        y = df.pop('target')
        dataset = df.pop('dataset')[0]
        df.columns = range(len(df.columns))
        X = df
        self._log_experiment_metadata(dataset)
        start_time = time.time()
        for xi, yi in tqdm(stream.iter_pandas(X, y)):
            pred, probas = self._stream_pipeline.handle(xi, yi)
            y_pred = int(pred)
            eval_report = self._stream_evaluation.handle(yi, y_pred)
            self._log_batch_results(eval_report)
        end_time = time.time()
        self._log_last_results()
        self._perf_logger.log_time(start_time, end_time)
        
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

    def get_params(self) -> dict:
        return {
            'type': self.__class__.__name__,
            'stream_pipeline': self._stream_pipeline.get_params(),
            'pipeline_evaluation': self._stream_evaluation.get_params()
        }

    def _log_experiment_metadata(self, dataset_name: str):
        params = {'dataset_name': dataset_name}
        params.update(self.get_params())
        self._logger.log_dict_as_json(params)
