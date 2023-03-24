import pandas as pd
from typing import List
import numpy as np



class HistoryManager:

    def __init__(self, n_flush_clock: int = 100_000, n_to_stay: int = 90_000, y_dtype=np.uint8):
        self._counter: int = 0
        self._x_history: List = []
        self._y_history: List = []
        self._idx_history: List = []
        self._prediction_history: List = []
        self._in_drift_history: List[List[int]] = []
        self._out_drift_history: List[List[int]] = []
        self._last_retraining: int = None
        self._n_flush_clock: int = n_flush_clock
        self._n_to_stay: int = n_to_stay
        self._first_index: int = 0
        self._y_dtype = y_dtype

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
        if self._counter % self._n_flush_clock == 0 and self._counter != 0:
            self._flush_x_history()
            self._y_history = self.y_history.iloc[-(self._n_to_stay):].to_list()
            self._prediction_history = self.prediction_history[-(self._n_to_stay):].to_list()

    def _flush_x_history(self):
        n, _ = self.x_history.shape
        x_flushed = self.x_history.iloc[-(self._n_to_stay):, :]
        self._first_index = x_flushed.index.min()
        self._x_history = x_flushed.to_dict('records')

    @property
    def counter(self) -> int:
        return self._counter
    
    @property
    def x_history(self) -> pd.DataFrame:
        df = pd.DataFrame(self._x_history)
        df.index = range(self._first_index, len(df) + self._first_index)
        return df

    @property
    def y_history(self) -> pd.Series:
        y = pd.Series(self._y_history, dtype=self._y_dtype)
        y.index = range(self._first_index, len(y) + self._first_index)
        return y

    @property
    def prediction_history(self) -> pd.Series:
        y_pred = pd.Series(self._prediction_history, dtype=self._y_dtype)
        y_pred.index = range(self._first_index, len(y_pred) + self._first_index)
        return y_pred

    @property
    def drift_history(self) -> List[List[int]]:
        return self._drift_history
