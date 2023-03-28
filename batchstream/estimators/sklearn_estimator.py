import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from .base.batch_model_estimator import BatchModelEstimator



class SklearnEstimator(BatchModelEstimator):
    
    def __init__(
            self,
            sklearn_estimator: Pipeline,
            **hyperparams_kwargs
        ):
        super().__init__(sklearn_estimator)
        self._hyperparams_kwargs = hyperparams_kwargs

    def retrain(self, X_retrain, y_retrain) -> Pipeline:
        cloned_model = clone(self.batch_model)
        if self._hyperparams_kwargs == None or len(self._hyperparams_kwargs) == 0:
            return cloned_model.fit(X_retrain, y_retrain)
        return self._tune_hyperparams(cloned_model, X_retrain, y_retrain)

    def first_fit(self, X_train, y_train) -> None:
        if self._hyperparams_kwargs is None or len(self._hyperparams_kwargs) == 0:
            self.batch_model.fit(X_train, y_train)
            return
        return self._tune_hyperparams(self._batch_model, X_train, y_train)

    def predict(self, X) -> pd.Series:
        return pd.Series(self.batch_model.predict(X))

    def _tune_hyperparams(self, model: Pipeline, X_train, y_train) -> BatchModelEstimator:
        search = GridSearchCV(model, **self._hyperparams_kwargs)
        search.fit(X_train, y_train)
        return search.best_estimator_

    def get_name(self) -> str:
        return "TO DO"
    
    def get_params(self) -> dict:
        params = {'type': self.__class__.__name__}
        params.update({'sklearn_estimator': self.batch_model.get_params()})
        params.update({'hyperparams': self._hyperparams_kwargs})
        return params
        