import pytest
import pandas as pd
from batchstream.utils.logging.performance_logger import PerformanceEvalLogger



# arrange
@pytest.fixture
def test_exp_name():
    return 'test_experiment'

@pytest.fixture
def actual_report_artifact(test_exp_name):
    return pd.read_csv(f'./out/{test_exp_name}/{test_exp_name}_performance_eval_report.csv')

@pytest.fixture
def actual_log_rows_number(test_exp_name):
    with open(f"./log/{test_exp_name}/{test_exp_name}_performance_eval.log", 'r') as fp:
        for count, _ in enumerate(fp):
            pass
    return count + 1

@pytest.fixture
def eval_reports_batch_1():
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
def eval_reports_batch_2():
    return [
        {'accuracy': 0.75, 'roc_auc': -0.0},
        {'accuracy': 0.71, 'roc_auc': 0.5},
        {'accuracy': 0.67, 'roc_auc': 0.5},
        {'accuracy': 0.6, 'roc_auc': 0.5},
        {'accuracy': 0.5, 'roc_auc': 0.67},
        {'accuracy': 0.4, 'roc_auc': 0.67},
        {'accuracy': 0.33, 'roc_auc': 0.75},
        {'accuracy': 0.2, 'roc_auc': 0.75}
    ]

@pytest.fixture
def expected_report_artifact(eval_reports_batch_1, eval_reports_batch_2):
    return pd.concat([pd.DataFrame(eval_reports_batch_1), pd.DataFrame(eval_reports_batch_2)]).reset_index()

@pytest.fixture
def expected_number_rows():
    return 4

@pytest.fixture
def perf_logger(test_exp_name):
    return PerformanceEvalLogger(experiment_id=test_exp_name)

# act
@pytest.fixture
def log(perf_logger: PerformanceEvalLogger, eval_reports_batch_1, eval_reports_batch_2):
    perf_logger.log_info("START logging performance_evaluation.")
    perf_logger.log_eval_report(eval_reports_batch_1)
    perf_logger.log_info("First batch logged.")
    perf_logger.log_eval_report(eval_reports_batch_2)
    perf_logger.log_info("Second batch logged.")
    perf_logger.log_info("END logging performance_evaluation.")

@pytest.fixture
def results(log, actual_report_artifact, actual_log_rows_number):
    return actual_report_artifact, actual_log_rows_number

# assert
def test_performance_logging(results, expected_report_artifact, expected_number_rows):
    actual_report_artifact: pd.DataFrame = results[0]
    actual_log_rows_number: int = results[1]

    pd.testing.assert_frame_equal(actual_report_artifact, expected_report_artifact)
    assert actual_log_rows_number == expected_number_rows
