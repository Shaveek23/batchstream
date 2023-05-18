from river import compose
from typing import List, Tuple
from ..base.stream_pipeline import StreamPipeline



class CombinationPipeline(StreamPipeline):
    
    def __init__(self, river_pipeline: compose.Pipeline):
        self.online_model = river_pipeline

    def handle(self, x, y) -> Tuple[int, List[float]]:
        prediction = self.online_model.predict_one(x)
        probabilities = self.online_model.predict_proba_one(x)
        self.online_model.learn_one(x, y)
        return prediction, probabilities

    def get_name(self):
        return super().get_name() # TO DO
    
    def get_params(self) -> dict:
        return {
            'type': self.__class__.__name__,
            'river_pipeline': self.online_model._get_params()
        }
