import pandas as pd
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from evidently.tests import *
from batchstream.history.base.history_manager import HistoryManager
from batchstream.monitoring.pipeline.model_monitoring_pipeline import ModelMonitoringPipeline
from batchstream.monitoring.pipeline.steps.batch.evidently_monitoring_step import EvidentlyMonitoringStep
from sklearn.datasets import load_breast_cancer
from batchstream.utils.logging.base.logger_factory import LoggerFactory
from batchstream.batch_monitoring_strategy.dummy_monitoring_strategy import DummyMonitoringStrategy
from batchstream.retraining_strategy.dummy_retraining_strategy import DummyRetrainingStrategy 
from batchstream.model_comparers.batch_comparer import BatchModelComparer
from batchstream.model_comparers.shadow_comparer import ShadowOnlineComparer
from batchstream.pipelines.batch.batch_pipeline import BatchPipeline
from batchstream.estimators.sklearn_estimator import SklearnEstimator
from batchstream.detectors.base.detector import DriftDetector
from batchstream.experiment.experiment import StreamExperiment
from river.metrics import Accuracy, ROCAUC
from batchstream.evaluation.model_evaluation_pipeline import ModelEvaluationPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline



def main():
    history = HistoryManager()
    logger_factory = LoggerFactory('test-2218')


    ### INPUT DRIFT DETECTION
    # Detector 1.1 - Data Drift
    data_drift_test_suite = TestSuite(tests=[
    DataDriftTestPreset(),
    ])
    d1 = DummyMonitoringStrategy(n_curr=120, n_ref=120)
    ev1 = EvidentlyMonitoringStep(data_drift_test_suite, d1, logger_factory, min_instances=240, clock=120, name='data_drift_eval')

    # Detector 1.2 - Target Drift
    target_drift = TestSuite(tests=[
        TestColumnDrift(column_name='target'),
    ])
    d2 = DummyMonitoringStrategy(n_curr=120, n_ref=120, type='target')
    ev2 = EvidentlyMonitoringStep(target_drift, d2, logger_factory, min_instances=240, clock=120, name='target_drift_eval')

    input_monitoring = ModelMonitoringPipeline([(ev1._name, ev1), (ev2._name, ev2)])
    input_drift_retraining_strategy = DummyRetrainingStrategy(n_last_retrain=120, n_last_test=0)
    input_detector = DriftDetector(input_monitoring, input_drift_retraining_strategy)
    ###




    ### OUTPUT (PERFORMANCE) DRIFT DETECTION
    # Detector 2.1 - Performance Drift

    performance_drift = TestSuite(tests=[
        TestPrecisionScore(),
        TestRecallScore(),
        TestF1Score(),
        TestAccuracyScore()
    ])
    d3 = DummyMonitoringStrategy(n_curr=120, n_ref=120, type='prediction')
    ev3 = EvidentlyMonitoringStep(performance_drift, d3, logger_factory, min_instances=360, clock=120, name='performance_drift_eval')

    output_monitoring = ModelMonitoringPipeline([(ev3._name, ev3)])
    output_drift_retraining_strategy = DummyRetrainingStrategy(n_last_retrain=120, n_last_test=0)
    output_detector = DriftDetector(output_monitoring, output_drift_retraining_strategy)
    ###

    ### Models comparison (after retraining)
    #model_comparer = BatchModelComparer()
    model_comparer = ShadowOnlineComparer(n_online=20)
    ###


    ### Model's Performance Evaluation
    acc = Accuracy()
    roc_auc = ROCAUC()
    eval_pipe = ModelEvaluationPipeline(metric_steps=[
        ('accuracy', acc),
        ('roc_auc', roc_auc)
    ])
    ###


    ### Model composition
    sklearn_batch_classifier = SklearnEstimator(Pipeline([('rf', RandomForestClassifier())]))
    batch_pipeline = BatchPipeline(
        sklearn_batch_classifier,
        input_drift_detector=input_detector,
        output_drift_detector=output_detector,
        history=history,
        logger_factory=logger_factory,
        model_comparer=model_comparer,
        min_samples_retrain=120,
        min_samples_first_fit=240
    )
    ###

    ### Experiment
    experiment = StreamExperiment(batch_pipeline, eval_pipe, logger_factory)

    X, Y = load_breast_cancer(return_X_y=True)
    df = pd.DataFrame(X)
    df['target'] = Y

    experiment.run(df)

if __name__ == "__main__":
    main()
