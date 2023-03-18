from .base.model_comparer import ModelOnlineComparer
from typing import List, Tuple



class ShadowOnlineComparer(ModelOnlineComparer):

    def __init__(self, n_online=10):
        super().__init__(self, n_online)

    def _is_new_better_than_old_online(self, x, y:int, old_model_prediction: int) -> Tuple(bool, object):
        if self._new_model is None:
            return False, None    
        if self._counter < self._n_online:
            self._counter += 1
            self._handle(x, y, old_model_prediction)
            return False, None
        res = self._make_decision()
        return res, self._new_model if res else self._old_model

    def _handle(self, x, y: int, old_model_prediction: int) -> int:
        self._y_true.append(y)
        y_pred_new = self._new_model.predict(x)
        self._predictions_new.append(y_pred_new)
        self._predictions_old.append(old_model_prediction)

    def _make_decision(self):
        return self._predictions_new == self._predictions_old # TO DO: kiedy new better than old
