import pandas as pd
from pipelines.base.stream_pipeline import StreamPipeline
from evaluation.base.model_evaluation import ModelEvaluation



class StreamExperiment:

    def __init__(
        self, # TO DO: logging
        stream_pipeline: StreamPipeline,
        pipeline_evaluation: ModelEvaluation
        ):
        pass

    def run(self, df: pd.DataFrame):
        for x, y in df: # TO DO
            y_pred = stream_pipeline.handle(x, y)
            pipeline_evaluation.handle(y, y_pred)
            