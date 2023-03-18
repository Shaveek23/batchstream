from abc import ABC, abstractmethod
import pandas as pd
from typing import List



class HistoryManager(ABC):

    def __init__(self):
        self._counter: int = 0
        self._x_history: List = None
        self._y_history: List = None
        self._prediction_history: List = None
        self._in_drift_history: List[List[int]] = None
        self._out_drift_history: List[List[int]] = None
        self._last_retraining: int = None

    @property
    def counter(self) -> int:
        return self._counter
    
    def update_history_x(self, x):
        self._x_history.append(x)
        self.increment_counter()

    def update_history_y(self, y):
        self._y_history.append(y)

    def update_retraining_info(self, drift_iter: int, detector_idx: int, type: str='out'):
        if type == 'in': 
            self._in_drift_history[detector_idx].append(drift_iter)
        else:
            self._out_drift_history[detector_idx].append(drift_iter)
        self._last_retraining = drift_iter

    def update_predictions(self, pred):
        self._prediction_history.append(pred)
    
    def increment_counter(self, n: int=1):
        self._counter += n

    @property
    def x_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._x_history)

    @property
    def y_history(self) -> pd.Series:
        return pd.Series(self._y_history)

    @property
    def prediction_history(self) -> pd.Series:
        return  pd.Series(self._prediction_history)

    @property
    def drift_history(self) -> List[List[int]]:
        return self._drift_history

    @abstractmethod
    def flush(self):
        pass
