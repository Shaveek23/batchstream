import pandas as pd
from typing import Tuple, Dict
from ..history.base.history_manager import HistoryManager
from .base.retraining_strategy import RetrainingStrategy



class SimpleRetrainingStrategy(RetrainingStrategy):

    def __init__(self, n_last_retrain: int=20, n_last_test: int=10):
        self._n_last_retrain = n_last_retrain
        self._n_last_test = n_last_test
    
    def get_retraining_data(self, history: HistoryManager) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        if self._n_last_test == None or self._n_last_test == 0:
            return  history.x_history.iloc[:-1, :].iloc[-self._n_last_retrain:, :], history.y_history.iloc[-self._n_last_retrain:], {}
        return history.x_history.iloc[:-1, :].iloc[-self._n_last_retrain:-self._n_last_test, :], history.y_history.iloc[-self._n_last_retrain:-self._n_last_test], {}

    def get_retest_data(self, history: HistoryManager) -> Tuple[pd.DataFrame, pd.Series]:
        if self._n_last_test == None or self._n_last_test == 0:
            return None, None
        return history.x_history.iloc[:-1, :].iloc[-self._n_last_test:, :], history.y_history.iloc[-self._n_last_test:]
    
    def get_params(self) -> dict:
        return {
            'type': self.__class__.__name__,
            'n_last_retrain': self._n_last_retrain,
            'n_last_test': self._n_last_test
        }
    