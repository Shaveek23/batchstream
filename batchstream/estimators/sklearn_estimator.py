from ..base.batch_model_estimator import BatchModelEstimator
import pandas as pd
from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV



class SklearnEstimator(BatchModelEstimator):
    
    def __init__(
            self,
            sklearn_estimator: Pipeline,
            **hyperparams_kwargs
        ):
        super().__init__(sklearn_estimator)
        self._hyperparams_kwargs = hyperparams_kwargs

    def handle(self, x, y: int=None) -> int:
        return self.batch_model.predict(x)

    def get_name(self) -> str:
        return "TO DO"

    def retrain(self, X_retrain, y_retrain) -> BatchModelEstimator:
        if self._hyperparams_kwargs == None or len(self._hyperparams_kwargs) == 0:
            self.batch_model.fit(X_retrain, y_retrain)
            return
        self.batch_model = self._tune_hyperparams(X_retrain, y_retrain)

    def first_fit(self, X_train, y_train) -> None:
        if self._hyperparams_kwargs is None or len(self._hyperparams_kwargs) == 0:
            self.batch_model.fit(X_train, y_train)
            return
        self.batch_model = self._tune_hyperparams(X_train, y_train)

    def predict(self, X) -> pd.Series:
        return pd.Series(self.batch_model.predict(X))

    def _tune_hyperparams(self, X_train, y_train) -> BatchModelEstimator:
        search = GridSearchCV(self.batch_model, **self._hyperparams_kwargs)
        search.fit(X_train, y_train)
        return search.best_estimator_
