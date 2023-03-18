from abc import ABC, abstractmethod
from history.base.history_manager import HistoryManager
from typing import Dict, Tuple
import pandas as pd


class RetrainingStrategy(ABC):
    
    @abstractmethod
    def get_retraining_data(history: HistoryManager) -> Tuple(pd.DataFrame, pd.Series, Dict):
        pass

    @abstractmethod
    def get_retest_data(history: HistoryManager) -> Tuple(pd.DataFrame, pd.Series):
        pass
