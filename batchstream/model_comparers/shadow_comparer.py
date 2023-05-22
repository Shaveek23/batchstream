from .base.model_comparer import ModelOnlineComparer
from typing import Tuple
import math
from sklearn.metrics import f1_score
from river.utils import dict2numpy




class ShadowOnlineComparer(ModelOnlineComparer):

    def __init__(self, n_online: int=500, is_hoeffding_bound: bool=False, hoeffding_delta: float=1e-7):
        super().__init__(n_online)
        self._hoeffding_delta = hoeffding_delta
        self._is_hoeff_bound = is_hoeffding_bound

    def _is_new_better_than_old_online(self, x, y:int, old_model_prediction: int) -> Tuple[bool, object]:
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
        y_pred_new = self._new_model.predict(dict2numpy(x).reshape(1, -1))[0]
        self._predictions_new.append(y_pred_new)
        self._predictions_old.append(old_model_prediction)

    def _make_decision(self):
        f1_new = f1_score(self._y_true, self._predictions_new, average='macro')
        f1_old = f1_score( self._y_true, self._predictions_old, average='macro')
        print(f"comparison: {f1_new - f1_old}")
        epsilon = 0.0
        if self._is_hoeff_bound:
            epsilon = self._hoeffding_bound(1, confidence=self._hoeffding_delta, n=len(self._predictions_new))
        return (f1_new - f1_old) > epsilon

    @staticmethod
    def _hoeffding_bound(range_val, confidence, n):
        r"""Compute the Hoeffding bound, used to decide how many samples are necessary at each
        node.

        Notes
        -----
        The Hoeffding bound is defined as:

        $\\epsilon = \\sqrt{\\frac{R^2\\ln(1/\\delta))}{2n}}$

        where:

        $\\epsilon$: Hoeffding bound.
        $R$: Range of a random variable. For a probability the range is 1, and for an
        information gain the range is log *c*, where *c* is the number of classes.
        $\\delta$: Confidence. 1 minus the desired probability of choosing the correct
        attribute at any given node.
        $n$: Number of samples.

        Parameters
        ----------
        range_val
            Range value.
        confidence
            Confidence of choosing the correct attribute.
        n
            Number of processed samples.
        """
        return math.sqrt((range_val * range_val * math.log(1.0 / confidence)) / (2.0 * n))

    
    def get_params(self) -> dict:
        return {
            'type': self.__class__.__name__,
            'n_online': self._n_online,
            'is_hoeff_bound': self._is_hoeff_bound,
            'hoeffding_delta': self._hoeffding_delta,
            'metric': 'F1 macro score'
        }
    