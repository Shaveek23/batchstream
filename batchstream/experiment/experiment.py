import pandas as pd
from pipelines.base.stream_pipeline import StreamPipeline
from evaluation.base.model_evaluation import ModelEvaluation
from river import stream


class StreamExperiment:

    def __init__(
            self, # TO DO: logging
            stream_pipeline: StreamPipeline,
            pipeline_evaluation: ModelEvaluation
        ):
        self._stream_pipeline = stream_pipeline
        self._stream_evaluation = pipeline_evaluation

    def run(self, df: pd.DataFrame):
        y = df.pop('target')
        X = df
        for xi, yi in stream.iter_pandas(X, y):
            y_pred = int(self._stream_pipeline.handle(xi, yi))
            self._pipeline_evaluation.handle(yi, y_pred)
            #TO DO: log results
            