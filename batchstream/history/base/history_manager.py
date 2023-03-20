import pandas as pd
from typing import List



class HistoryManager:

    def __init__(self, n_flush: int = 100_000):
        self._counter: int = 0
        self._x_history: List = None
        self._y_history: List = None
        self._prediction_history: List = None
        self._in_drift_history: List[List[int]] = None
        self._out_drift_history: List[List[int]] = None
        self._last_retraining: int = None
        self._n_flush: int = n_flush

    def update_history_x(self, x):
        self._x_history.append(x)

    def update_history_y(self, y):
        self._y_history.append(y)
        self.increment_counter()
        self.flush()

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

    def flush(self):
        if self._counter % self._n_flush == 0 and self._counter != 0:
            self._x_history = self._x_history[self._n_flush:]
            self._y_history = self._y_history[self._n_flush:]
            self._prediction_history = self._prediction_history[self._n_flush:]

    @property
    def counter(self) -> int:
        return self._counter
    
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
