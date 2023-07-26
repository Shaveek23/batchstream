import pandas as pd
from typing import List, Dict
import numpy as np



class HistoryManager:

    def __init__(self, n_flush_clock: int = 20_000, n_to_stay: int = 15_000, y_dtype=np.int8):
        self._counter: int = 0
        self._x_history: List = []
        self._y_history: List = []
        self._idx_history: List = []
        self._prediction_history: List = []
        self._in_drift_history: Dict[int, List[int]] = {}
        self._out_drift_history: Dict[int, List[int]] = {}
        self._last_retraining: int = None
        self._replacement_history: List[int] = []
        self._n_flush_clock: int = n_flush_clock
        self._n_to_stay: int = n_to_stay
        self._first_index: int = 0
        self._y_dtype = y_dtype

    def update_history_x(self, x):
        self._x_history.append(x)

    def update_history_y_and_pred(self, y, pred):
        self._prediction_history.append(pred)
        self._y_history.append(y)
        self.increment_counter()
        self.flush()

    def update_replacement_history(self, idx: int):
        self._replacement_history.append(idx)

    def get_last_replacement_idx(self):
        return self._replacement_history[-1]

    def update_retraining_info(self, drift_iter: int, detector_idx: int, type: str='out'):
        if type == 'in': 
            d = self._in_drift_history
        else:
            d = self._out_drift_history
        if detector_idx not in d:
            d.update({detector_idx: []})
        d[detector_idx].append(drift_iter)
        self._last_retraining = drift_iter
    
    def increment_counter(self, n: int=1):
        self._counter += n

    def flush(self):
        if self._counter % self._n_flush_clock == 0 and self._counter != 0:
            self._flush_x_history()
            self._y_history = self._y_history[-(self._n_to_stay):]
            self._prediction_history = self._prediction_history[-(self._n_to_stay):]

    def _flush_x_history(self):
        self._first_index = self._counter - self._n_to_stay
        self._x_history = self._x_history[-(self._n_to_stay):]

    def get_x_history_as_pd(self) -> pd.DataFrame:
        df = pd.DataFrame(self._x_history)
        df.index = range(self._first_index, len(df) + self._first_index)
        return df

    def get_y_history_as_pd(self) -> pd.Series:
        y = pd.Series(self._y_history, dtype=self._y_dtype)
        y.index = range(self._first_index, len(y) + self._first_index)
        return y

    def get_prediction_history_as_pd(self) -> pd.Series:
        y_pred = pd.Series(self._prediction_history, dtype=self._y_dtype)
        y_pred.index = range(self._first_index, len(y_pred) + self._first_index)
        return y_pred
    
    @property
    def counter(self) -> int:
        return self._counter
    
    @property
    def x_history(self) -> List[Dict]:
        return self._x_history

    @property
    def y_history(self) -> List[Dict]:
        return self._y_history

    @property
    def prediction_history(self) -> List:
        return self._prediction_history

    @property
    def drift_history(self) -> List[List[int]]:
        return self._drift_history
    
    @property
    def replacement_history(self) -> List[int]:
        return self._replacement_history
    
    
    def get_params(self):
        return {
            'type': self.__class__.__name__,
            'n_flush_clock': self._n_flush_clock,
            'n_to_stay': self._n_to_stay,
            'y_dtype': str(self._y_dtype)
        }
