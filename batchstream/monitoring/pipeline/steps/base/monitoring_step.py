from abc import ABC, abstractmethod
from typing import List, Dict



class MonitoringStep(ABC):

    @abstractmethod
    def monitor(self, x_history: List, y_history: List[int], prediction_history: List[int], drift_history: List[int]) -> dict:
        pass
