from __future__ import annotations
from abc import abstractmethod, ABCMeta
from model_estimator import ModelEstimator
import pandas as pd


class BatchModelEstimator(ModelEstimator, metadata=ABCMeta):

    def __init__(self, batch_model):
        self._batch_model = batch_model

    @abstractmethod
    def retrain(self, X_retrain, y_retrain, **retrain_kwargs) -> BatchModelEstimator:
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
        