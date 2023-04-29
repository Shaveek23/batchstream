from abc import ABC, abstractmethod
from ...history.base.history_manager import HistoryManager



class DriftMonitoring(ABC):

    @abstractmethod
    def monitor(self, history: HistoryManager) -> bool:
        pass

    @abstractmethod 
    def get_name(self):
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass
