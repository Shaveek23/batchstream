from river import compose
from typing import List
from ..base.stream_pipeline import StreamPipeline



class ModelRiverPipeline(StreamPipeline):
    
    def __init__(self, river_pipeline: compose.Pipeline):
        self.online_model = river_pipeline

    def handle(self, x, y):
        prediction = self.online_model.predict_one(x)
        self.online_model.learn_one(x, y)
        return prediction

    def get_name(self):
        return super().get_name() # TO DO
    
    def get_params(self) -> dict:
        return {
            'type': self.__class__.__name__,
            'river_pipeline': self.online_model._get_params()
        }
