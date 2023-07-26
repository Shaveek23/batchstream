import pandas as pd
from typing import Tuple, Dict
from ..history.base.history_manager import HistoryManager
from .base.retraining_strategy import RetrainingStrategy



class FromLastReplacementRetrainingStrategy(RetrainingStrategy):

    def __init__(self):
        pass

    def get_retraining_data(self, history: HistoryManager) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        if len(history.x_history) != len(history.y_history):
            x_history = history.x_history[:-1]
        else:
            x_history = history.x_history
       
        X = x_history[history.get_last_replacement_idx():]
        Y = history.y_history[history.get_last_replacement_idx():]
        return pd.DataFrame(X), pd.Series(Y), {}

    def get_retest_data(self, history: HistoryManager) -> Tuple[pd.DataFrame, pd.Series]:
       return [pd.DataFrame(), pd.DataFrame()]
    
    def get_params(self) -> dict:
        return {
            'type': self.__class__.__name__,
        }