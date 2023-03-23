from abc import abstractmethod, ABC
import pandas as pd



class BatchModelEstimator(ABC):

    def __init__(self, batch_model: object):
        self._batch_model = batch_model

    @abstractmethod
    def retrain(self, X_retrain, y_retrain, **retrain_kwargs) -> object:
        pass

    @abstractmethod
    def first_fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X) -> pd.Series:
        pass

    @property
    def batch_model(self):
        return self._batch_model
    
    @batch_model.setter
    def _set_batch_model(self, value):
        self._batch_model = value
        