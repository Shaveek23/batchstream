from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from typing import Tuple, List



class ModelComparer(ABC):

    @abstractproperty
    def detector_idx(self):
        pass
    
    @abstractproperty 
    def drift_iter(self):
        pass
    
    @abstractproperty
    def detector_type(self):
        pass

    @abstractproperty
    def is_online_in_progress(self):
        pass
 
    @abstractmethod
    def is_new_better_than_old_online(self, x, y: int, old_model_prediction: int) -> Tuple[bool, object, int, int, str]:
        pass
    
    @abstractmethod
    def is_new_better_than_old_offline(self, new_model, old_model, X, y) -> Tuple[bool, object]:
        pass
     
    @abstractmethod
    def trigger_online_comparing(self, new_model, old_model, drift_iter:int, detector_idx: int, detector_type: str) -> None:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass 


class ModelOnlineComparer(ModelComparer):

    def __init__(self, n_online:int):
        super().__init__()
        self._n_online = n_online
        self._new_model = None
        self._old_model = None
        self._predictions_old: List[int] = []
        self._predictions_new: List[int] = []
        self._y_true: List[int] = []
        self._counter = 0
        self._drift_iter: int | None = None
        self._detector_idx: int | None = None
        self._detector_type: int | None = None
        self._is_online_in_progress: bool = False

    @property
    def detector_idx(self):
        return self._detector_idx
    
    @property 
    def drift_iter(self):
        return self._drift_iter
    
    @property
    def detector_type(self):
        return self.detector_type
    
    @property
    def is_online_in_progress(self):
        return self._is_online_in_progress


    def is_new_better_than_old_online(self, x, y:int, old_model_prediction: int) -> Tuple[bool, object, int, int, str]:
        if self._n_online is None or self._new_model is None or self._old_model is None:
            return False, None, None, None, None
        is_new_better, better_model = self._is_new_better_than_old_online(x, y, old_model_prediction)
        if better_model != None:
            drift_iter = self._drift_iter
            detector_idx = self._detector_idx
            detector_type = self._detector_type
            self._clean_and_stop_comparing()
            return is_new_better, better_model, drift_iter, detector_idx, detector_type
        return False, None, None, None, None
    
    def _clean_and_stop_comparing(self):
        self._new_model = None
        self._old_model = None
        self._predictions_old = []
        self._predictions_new = []
        self._y_true = []
        self._counter = 0
        self._drift_iter = None
        self._detector_idx = None
        self._detector_type = None
        self._is_online_in_progress = False

    @abstractmethod
    def _is_new_better_than_old_online(x, y: int, old_model_prediction: int) -> Tuple[bool, object]:
        '''
            x: current features
            y: a current true label
            This function must be overridden in all concrete online comparer classes.
            
            It needs to return a tuple:
                * (`True`, `new_model`) - when the online evaluation is over and a `new_model` won.
                * (`False`, `old_model`) - when the online evaluation is over and an `old_model` won.
                * (`False`, `None`) - when the online evaluation is still in progress or has not started yet. 

            All superclass fields such as counter, new_model, old_model, predictions are reset when the second field 
            in the result tuple is not None.
        '''
        pass
        
    def trigger_online_comparing(self, new_model, old_model, drift_iter:int, detector_idx: int, detector_type: str) -> None:
        if self._is_online_in_progress: return
        self._clean_and_stop_comparing()
        self._new_model = new_model
        self._old_model = old_model
        self._drift_iter = drift_iter
        self._detector_idx = detector_idx
        self._detector_type = detector_type
        self._is_online_in_progress = True

    # Make offline functionalities transparent
    def is_new_better_than_old_offline(self, new_model, old_model, X, y) -> Tuple[bool, object]:
        return False, None


class ModelOfflineComparer(ModelComparer):

    def is_new_better_than_old_offline(self, new_model, old_model, X, y) -> Tuple[bool, object]:
        if self._is_new_better_than_old_offline(new_model, old_model, X, y):
            return True, new_model
        else: 
            return False, old_model
        
    @abstractmethod
    def _is_new_better_than_old_offline(self, new_model, old_model, X, y) -> bool:
        pass

    # Make online functionalities transparent
    @property
    def detector_idx(self):
        return None
    
    @property 
    def drift_iter(self):
        return None
    
    @property
    def detector_type(self):
        return None
    
    @property
    def is_online_in_progress(self):
        return False

    def is_new_better_than_old_online(self, x, y, old_model_prediction: int) -> Tuple[bool, object]:
        return False, None
    
    def trigger_online_comparing(self, new_model, old_model, drift_iter:int, detector_idx: int, detector_type: str) -> None:
        return
        