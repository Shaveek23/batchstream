from abc import ABC, abstractmethod



class ModelEvaluation(ABC):
    
    @abstractmethod
    def handle(self, y_true, y_predict):
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass
    