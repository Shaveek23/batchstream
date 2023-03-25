import pandas as pd
from typing import Tuple, Dict
from ..history.base.history_manager import HistoryManager
from .base.retraining_strategy import RetrainingStrategy



class DummyRetrainingStrategy(RetrainingStrategy):

    def __init__(self, n_last_retrain: int=20, n_last_test: int=10):
        self._n_last_retrain = n_last_retrain
        self._n_last_test = n_last_test
    
    def get_retraining_data(self, history: HistoryManager) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        return history.x_history.iloc[:-1, :].iloc[-self._n_last_retrain:-self._n_last_test, :], history.y_history.iloc[-self._n_last_retrain:-self._n_last_test], {}

    def get_retest_data(self, history: HistoryManager) -> Tuple[pd.DataFrame, pd.Series]:
        return history.x_history.iloc[:-1, :].iloc[-self._n_last_test:, :], history.y_history.iloc[-self._n_last_test:]
    