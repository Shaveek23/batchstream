from abc import ABC, abstractmethod
from typing import Dict, Tuple
import pandas as pd
from ...history.base.history_manager import HistoryManager



class RetrainingStrategy(ABC):
    
    @abstractmethod
    def get_retraining_data(history: HistoryManager) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        pass

    @abstractmethod
    def get_retest_data(history: HistoryManager) -> Tuple[pd.DataFrame, pd.Series]:
        pass
