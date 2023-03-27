from .base.model_comparer import ModelOfflineComparer
from sklearn.metrics import f1_score



class BatchModelComparer(ModelOfflineComparer):

    def _is_new_better_than_old_offline(self, new_model, old_model, X, y) -> bool:
        y_pred_new = new_model.predict(X)
        y_pred_old = old_model.predict(X)
        return self._make_decision(y_pred_new, y_pred_old, y)

    def _make_decision(self, y_pred_new, y_pred_old, y):
        f1_new = f1_score(y_pred_new, y)
        f1_old = f1_score(y_pred_old, y)
        return f1_new >= f1_old
