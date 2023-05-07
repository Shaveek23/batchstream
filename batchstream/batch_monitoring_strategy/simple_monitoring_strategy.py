import pandas as pd
from batchstream.history.base.history_manager import HistoryManager
from .base.batch_monitoring_strategy import BatchMonitoringStrategy



class SimpleMonitoringStrategy(BatchMonitoringStrategy):

    def __init__(self, n_curr: int=400, n_ref=400, type='data'):
        self.n_curr = n_curr
        self.n_ref = n_ref
        self.type = type

    def get_ref_curr(self, history: HistoryManager):
        if self.type == 'data':
            df = history.get_x_history_as_pd()
            return df.iloc[-(self.n_curr + self.n_ref):-self.n_curr, :], df.iloc[-self.n_curr:, :]
        elif self.type == 'target':
            target = history.get_y_history_as_pd()
            ref = target.iloc[-(self.n_curr + self.n_ref):-self.n_curr]
            curr = target.iloc[-self.n_curr:] 
            return pd.DataFrame(ref, columns=['target']), pd.DataFrame(curr, columns=['target'])
        elif self.type == 'prediction':
            prediction = history.get_prediction_history_as_pd()
            ref = prediction.iloc[-(self.n_curr + self.n_ref):-self.n_curr]
            curr = prediction.iloc[-self.n_curr:] 
            return pd.DataFrame(ref, columns=['prediction']), pd.DataFrame(curr, columns=['prediction'])
        return None, None

    def get_params(self) -> dict:
        return {
            'type': self.__class__.__name__,
            'n_curr': self.n_curr,
            'n_ref': self.n_ref,
            'type': self._type
        }
    