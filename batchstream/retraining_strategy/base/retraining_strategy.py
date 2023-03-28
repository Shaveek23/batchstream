from abc import ABC, abstractmethod
from typing import Dict, Tuple
import pandas as pd
from ...history.base.history_manager import HistoryManager



class RetrainingStrategy(ABC):
    
    @abstractmethod
    def get_retraining_data(self, history: HistoryManager) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        pass

    @abstractmethod
    def get_retest_data(self, history: HistoryManager) -> Tuple[pd.DataFrame, pd.Series]:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass
