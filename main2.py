import pandas as pd
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from evidently.tests import *
from batchstream.history.base.history_manager import HistoryManager
from batchstream.monitoring.pipeline.drift_monitoring_pipeline import DriftMonitoringPipeline
from batchstream.monitoring.pipeline.steps.batch.evidently_monitoring_step import EvidentlyMonitoringStep
from sklearn.datasets import load_breast_cancer
from batchstream.utils.logging.base.logger_factory import LoggerFactory
from batchstream.batch_monitoring_strategy.simple_monitoring_strategy import SimpleMonitoringStrategy
from batchstream.retraining_strategy.simple_retraining_strategy import SimpleRetrainingStrategy 
from batchstream.model_comparers.shadow_comparer import ShadowOnlineComparer
from batchstream.pipelines.batch.batch_pipeline import BatchPipeline
from batchstream.estimators.sklearn_estimator import SklearnEstimator
from batchstream.drift_handlers.base.drift_handler import DriftHandler
from batchstream.experiment.experiment import StreamExperiment
from river.metrics import Accuracy, MacroF1, CohenKappa
from river.utils import Rolling
from batchstream.evaluation.river_evaluation_pipeline import RiverEvaluationPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from utils.read_data.covtype import get_covtype_dataset
from datetime import datetime


def main():
    name = 'test_covtype_rf_all'
    exp_name = f'{name}_{datetime.today().strftime("%Y%m%d_%H%M%S")}'
    history = HistoryManager()
    logger_factory = LoggerFactory(exp_name)


    ### INPUT DRIFT DETECTION
    # Detector 1.1 - Data Drift
    data_drift_test_suite = {'tests': [
    DataDriftTestPreset(),
    ]}
    d1 = SimpleMonitoringStrategy(n_curr=5000, n_ref=5000)
    ev1 = EvidentlyMonitoringStep(data_drift_test_suite, d1, logger_factory, min_instances=5000, clock=5000, name='data_drift_eval')

    # Detector 1.2 - Target Drift
    target_drift = {'tests': [
        TestColumnDrift(column_name='target'),
    ]}
    d2 = SimpleMonitoringStrategy(n_curr=5000, n_ref=5000, type='target')
    ev2 = EvidentlyMonitoringStep(target_drift, d2, logger_factory, min_instances=5000, clock=5000, name='target_drift_eval')

    input_monitoring = DriftMonitoringPipeline([(ev1.name, ev1), (ev2.name, ev2)])
    input_drift_retraining_strategy = SimpleRetrainingStrategy(n_last_retrain=5000, n_last_test=0)
    input_detector = DriftHandler(input_monitoring, input_drift_retraining_strategy)
    ###




    ### OUTPUT (PERFORMANCE) DRIFT DETECTION
    # Detector 2.1 - Performance Drift

    performance_drift = {'tests': [
        TestPrecisionScore(),
        TestRecallScore(),
        TestF1Score(),
        TestAccuracyScore()
    ]}
    d3 = SimpleMonitoringStrategy(n_curr=5000, n_ref=5000, type='prediction')
    ev3 = EvidentlyMonitoringStep(performance_drift, d3, logger_factory, min_instances=5000, clock=5000, name='performance_drift_eval')

    output_monitoring = DriftMonitoringPipeline([(ev3.name, ev3)])
    output_drift_retraining_strategy = SimpleRetrainingStrategy(n_last_retrain=5000, n_last_test=0)
    output_detector = DriftHandler(output_monitoring, output_drift_retraining_strategy)
    ###

    ### Models comparison (after retraining)
    #model_comparer = BatchModelComparer()
    model_comparer = ShadowOnlineComparer(n_online=100)
    ###


    ### Model's Performance Evaluation
    window_size = 1000
    eval_pipe = RiverEvaluationPipeline(metric_steps=[
        (f'acc_preq_{window_size}', Rolling(Accuracy(), window_size)),
        (f'macro_f1_preq_{window_size}', Rolling(MacroF1(), window_size)),
        (f'kappa_preq_{window_size}', Rolling(CohenKappa(), window_size)),
        ('acc', Accuracy()),
        ('f1_macro', MacroF1()),
        ('kappa', CohenKappa())
    ])
    ###


    ### Model composition
    sklearn_batch_classifier = SklearnEstimator(Pipeline([('rf', RandomForestClassifier())]))
    batch_pipeline = BatchPipeline(
        sklearn_batch_classifier,
        input_drift_handlers=input_detector,
        output_drift_handlers=output_detector,
        history=history,
        logger_factory=logger_factory,
        model_comparer=model_comparer,
        min_samples_retrain=5000,
        min_samples_first_fit=1000
    )
    ###

    ### Experiment
    experiment = StreamExperiment(batch_pipeline, eval_pipe, logger_factory)

    df = get_covtype_dataset()
    experiment.run(df)

if __name__ == "__main__":
    main()
