import pytest
from river.metrics import Accuracy, ROCAUC
from batchstream.evaluation.model_evaluation_pipeline import ModelEvaluationPipeline



# arrange
@pytest.fixture
def pred_true():
    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    y_pred = [0, 0, 0, 0, 1, 0, 1, 0]
    return zip(y_true, y_pred)

@pytest.fixture
def expected_results():
    return [
        {'accuracy': 0.0, 'roc_auc': -0.0},
        {'accuracy': 0.5, 'roc_auc': 0.5},
        {'accuracy': 0.33, 'roc_auc': 0.5},
        {'accuracy': 0.5, 'roc_auc': 0.5},
        {'accuracy': 0.6, 'roc_auc': 0.67},
        {'accuracy': 0.67, 'roc_auc': 0.67},
        {'accuracy': 0.71, 'roc_auc': 0.75},
        {'accuracy': 0.75, 'roc_auc': 0.75}
    ]

@pytest.fixture
def eval_pipe():
    acc = Accuracy()
    roc_auc = ROCAUC()
    eval_pipe = ModelEvaluationPipeline(metric_steps=[('accuracy', acc), ('roc_auc', roc_auc)])
    return eval_pipe

# act
@pytest.fixture
def actual_results(pred_true, eval_pipe):
    results = []
    for y_t, y_p in pred_true:
        x = eval_pipe.handle(y_t, y_p)
        results.append(x)

    return results

# assert
def test_model_evaluation_pipeline_handle(actual_results, expected_results):
    assert actual_results == expected_results
