from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from river.metrics import Accuracy, MacroF1, CohenKappa
from river.utils import Rolling
from batchstream.utils.concurrent import run_concurrent
from evidently.test_preset import DataDriftTestPreset
from evidently.tests import *
from batchstream.retraining_strategy.from_last_replacement_retraining_strategy import FromLastReplacementRetrainingStrategy
from batchstream.batch_monitoring_strategy.simple_monitoring_strategy import SimpleMonitoringStrategy
from batchstream.monitoring.pipeline.steps.batch.evidently_monitoring_step import EvidentlyMonitoringStep
from batchstream.drift_handlers.base.drift_handler import DriftHandler
from batchstream.estimators.sklearn_estimator import SklearnEstimator
from batchstream.evaluation.river_evaluation_pipeline import RiverEvaluationPipeline
from batchstream.experiment.experiment import StreamExperiment
from batchstream.history.base.history_manager import HistoryManager
from batchstream.model_comparers.shadow_comparer import ShadowOnlineComparer
from batchstream.monitoring.pipeline.drift_monitoring_pipeline import DriftMonitoringPipeline
from batchstream.pipelines.batch.batch_pipeline import BatchPipeline
from batchstream.retraining_strategy.simple_retraining_strategy import SimpleRetrainingStrategy
from batchstream.utils.logging.base.logger_factory import LoggerFactory
from utils.read_data.get_dataset import get_dataset



def get_demo_batch_experiment():
    '''
        A function returning a ready-to-use batch learning experiment using sklearn Random Forest.
        There are two input monitors (feature + target columns) combined into one detector with a simple retraining strategy,
        and one output monitor (performance) within a second detector with a retraining strategy that builds new model on data seen after the first replacement.
        Check the out directory for results after running the experiment.
    '''
    window_size = 1000
    eval_pipe = RiverEvaluationPipeline(metric_steps=[
        (f'acc_preq_{window_size}', Rolling(Accuracy(), window_size)),
        (f'macro_f1_preq_{window_size}', Rolling(MacroF1(), window_size)),
        (f'kappa_preq_{window_size}', Rolling(CohenKappa(), window_size)),
        ('acc', Accuracy()),
        ('f1_macro', MacroF1()),
        ('kappa', CohenKappa())
    ])
        
    logger_factory = LoggerFactory('rf_exp')
    history = HistoryManager()

    ### INPUT DRIFT DETECTION
    # Monitor 1.1 - Data Drift
    data_drift_test_suite = {'tests': [
        DataDriftTestPreset(stattest_threshold=0.043)
    ]}
    d1 = SimpleMonitoringStrategy(n_curr=5000, n_ref=5000)
    ev1 = EvidentlyMonitoringStep(data_drift_test_suite, d1, logger_factory,
    min_instances=2*5000, clock=5000, name='data_drift_eval')

    # Monitor 1.2 - Target Drift
    target_drift_tests = {'tests': [
        TestColumnDrift(column_name='target', stattest_threshold=0.043),
    ]}
    d2 = SimpleMonitoringStrategy(n_curr=5000, n_ref=5000, type='target')
    ev2 = EvidentlyMonitoringStep(target_drift_tests, d2, logger_factory, min_instances=2*5000, clock=5000, name='target_drift_eval')

    # Monitor 1.1. + Detector 1.2 with SimpleRetrainingStrategy when detectors detect drift
    input_monitoring = DriftMonitoringPipeline([(ev1.name, ev1), (ev2.name, ev2)])
    input_drift_retraining_strategy = SimpleRetrainingStrategy(n_last_retrain=5000, n_last_test=0)
    input_detector = DriftHandler(input_monitoring, input_drift_retraining_strategy)


    ### OUTPUT (PERFORMANCE) DRIFT DETECTION
    # Monitor 2.1 - Performance Drift

    performance_drift = {'tests': [
        TestPrecisionScore(),
        TestRecallScore(),
        TestF1Score(),
        TestAccuracyScore()
    ]}
    d3 = SimpleMonitoringStrategy(n_curr=5000, n_ref=5000, type='prediction')
    ev3 = EvidentlyMonitoringStep(performance_drift, d3, logger_factory, min_instances=2*5000, clock=5000, name='performance_drift_eval')

    output_monitoring = DriftMonitoringPipeline([(ev3.name, ev3)])
    output_drift_retraining_strategy = FromLastReplacementRetrainingStrategy()
    output_detector = DriftHandler(output_monitoring, output_drift_retraining_strategy)

    ### Models comparison (after retraining)
    model_comparer = ShadowOnlineComparer(n_online=100)

    ### Model composition
    sklearn_batch_classifier = SklearnEstimator(
    Pipeline([('rf', RandomForestClassifier())])
    )

    batch_pipeline = BatchPipeline(
        sklearn_batch_classifier,
        input_drift_handlers=input_detector,
        output_drift_handlers=output_detector,
        history=history,
        logger_factory=logger_factory,
        model_comparer=model_comparer,
        min_samples_retrain=5000,
        min_samples_first_fit=500
    )


    ### Experiment
    rf_experiment = StreamExperiment(batch_pipeline, eval_pipe, logger_factory)
    return rf_experiment


def main():
    dataset_name = 'ohio'
    df = get_dataset(dataset_name)
    exp_list = []
    exp1 = get_demo_batch_experiment()
    exp_list.append((exp1, df.copy(deep=True)))
    run_concurrent(exp_list)

if __name__ == "__main__":
    main()
