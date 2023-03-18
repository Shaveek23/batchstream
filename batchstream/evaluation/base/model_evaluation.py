from abc import ABC, abstractmethod



class ModelEvaluation(ABC):
    
    @abstractmethod
    def handle(self, y_true, y_predict):
        pass
    