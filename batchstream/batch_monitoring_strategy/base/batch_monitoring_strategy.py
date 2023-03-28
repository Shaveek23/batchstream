from abc import ABC, abstractmethod
from ...history.base.history_manager import HistoryManager



class BatchMonitoringStrategy(ABC):

    @abstractmethod
    def get_ref_curr(self, history: HistoryManager):
        pass
    
    @abstractmethod
    def get_params(self) -> dict:
        pass
