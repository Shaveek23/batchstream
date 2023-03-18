from monitoring.base.model_monitoring import ModelMonitoring
from retraining_strategy.base.retraining_strategy import RetrainingStrategy
from history.base.history_manager import HistoryManager
from models.base.batch_model_estimator import BatchModelEstimator
from typing import Tuple
import pandas as pd



class DriftDetector:

    def __init__(self, monitor: ModelMonitoring, retrain_strategy: RetrainingStrategy):
        self._monitor = monitor
        self._retrain_strategy = retrain_strategy

    def detect(self, history: HistoryManager) -> bool:
        return self._monitor.monitor(history)
    
    def react(self, batch_model: BatchModelEstimator, history: HistoryManager) -> Tuple(BatchModelEstimator, pd.DataFrame, pd.Series):
        X_retrain, y_retrain, retrain_kwargs = self._retrain_strategy.get_retraining_data(history)
        retrain_kwargs = {} if retrain_kwargs is None else retrain_kwargs
        retrained_model = batch_model.retrain(X_retrain, y_retrain, **retrain_kwargs)
        X_retest, y_retest = self._retrain_strategy.get_retest_data(history)
        return retrained_model, X_retest, y_retest
    