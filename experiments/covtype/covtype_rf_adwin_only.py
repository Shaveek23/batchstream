from batchstream.monitoring.pipeline.steps.online.river_monitoring_step import RiverMonitoringStep
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
from datetime import datetime
from river import drift
import uuid



def get_covtype_rf_adwin_only_exp(df, suffix, clock=5000, grace_period=5000, min_window_length=1000, n_online=100, windows_size=1000, n_first_fit=1000):
    prefix = str(uuid.uuid4())[:8]
    name = f'{prefix}_covtype_rf_adwin_only_{suffix}'
    exp_name = f'{name}_{datetime.today().strftime("%Y%m%d_%H%M%S")}'
    history = HistoryManager()
    logger_factory = LoggerFactory(exp_name)


    ### INPUT DRIFT DETECTION
    # Detector 1.1 - ADWIN
    
    adwins = []
    for col in df.columns:
        if col == 'dataset': continue
        if col == 'target': continue
        adwin = RiverMonitoringStep(col, drift.ADWIN(clock=clock, grace_period=grace_period, min_window_length=min_window_length), logger_factory)
        adwins.append(adwin)

    input_monitoring = DriftMonitoringPipeline([(a._name, a) for a in adwins])
    input_drift_retraining_strategy = SimpleRetrainingStrategy(n_last_retrain=clock, n_last_test=0)
    input_detector = DriftHandler(input_monitoring, input_drift_retraining_strategy)
    ###
   
    ### Models comparison (after retraining)
    model_comparer = ShadowOnlineComparer(n_online=n_online)
    ###


    ### Model's Performance Evaluation
    window_size = windows_size
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
        output_drift_handlers=None,
        history=history,
        logger_factory=logger_factory,
        model_comparer=model_comparer,
        min_samples_retrain=clock,
        min_samples_first_fit=n_first_fit
    )
    ###

    ### Experiment args
    experiment = StreamExperiment(batch_pipeline, eval_pipe, logger_factory)
    
    return experiment
