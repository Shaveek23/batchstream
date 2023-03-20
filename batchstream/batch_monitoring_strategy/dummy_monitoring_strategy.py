from batchstream.history.base.history_manager import HistoryManager
from .base.batch_monitoring_strategy import BatchMonitoringStrategy



class DummyMonitoringStrategy(BatchMonitoringStrategy):

    def __init__(self, n_curr: int=400, n_ref=400, type='data'):
        self.n_curr = n_curr
        self.n_ref = n_ref
        self.type = type

    def get_ref_curr(self, history: HistoryManager):
        if self.type == 'data':
            df = history.x_history
        else:
            df = history.y_history
        return df.iloc[-(self.n_curr + self.n_ref):-self.n_curr, :], df.iloc[-self.n_curr:, :]
