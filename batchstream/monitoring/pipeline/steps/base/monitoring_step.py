from abc import ABC, abstractmethod
from typing import List, Dict
from .....history.base.history_manager import HistoryManager



class MonitoringStep(ABC):

    @abstractmethod
    def monitor(self, history: HistoryManager) -> bool:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass
