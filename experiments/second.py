import uuid
from datetime import datetime

import pandas as pd
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.tests import *
from river import drift
from river.forest import ARFClassifier
from river.metrics import Accuracy, CohenKappa, MacroF1
from river.utils import Rolling
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from batchstream.batch_monitoring_strategy.simple_monitoring_strategy import \
    SimpleMonitoringStrategy
from batchstream.retraining_strategy.from_last_replacement_retraining_strategy import FromLastReplacementRetrainingStrategy
from batchstream.combine.majority_vote_combiner import MajorityVoteCombiner
from batchstream.drift_handlers.base.drift_handler import DriftHandler
from batchstream.estimators.sklearn_estimator import SklearnEstimator
from batchstream.evaluation.river_evaluation_pipeline import \
    RiverEvaluationPipeline
from batchstream.experiment.experiment import StreamExperiment
from batchstream.history.base.history_manager import HistoryManager
from batchstream.model_comparers.shadow_comparer import ShadowOnlineComparer
from batchstream.monitoring.pipeline.drift_monitoring_pipeline import \
    DriftMonitoringPipeline
from batchstream.monitoring.pipeline.steps.batch.evidently_monitoring_step import \
    EvidentlyMonitoringStep
from batchstream.monitoring.pipeline.steps.online.river_monitoring_step import \
    RiverMonitoringStep
from batchstream.pipelines.batch.batch_pipeline import BatchPipeline
from batchstream.pipelines.combine.combination_pipeline import \
    CombinationPipeline
from batchstream.pipelines.stream.model_river_pipeline import RiverPipeline
from batchstream.retraining_strategy.simple_retraining_strategy import \
    SimpleRetrainingStrategy
from batchstream.utils.logging.base.logger_factory import LoggerFactory
from batchstream.combine.majority_vote_combiner import MajorityVoteCombiner
from batchstream.combine.dynamic_switch_combiner import DynamicSwitchCombiner
from batchstream.combine.weighted_vote_combiner import WeightedVoteCombiner



def get_evidently_input_handlers(n_curr, n_ref, data_stattest_threshold, target_stattest_threshold, logger_factory, data_drift=True, target_drift=True):
    if not data_drift and not target_drift: return None
    
    ### INPUT DRIFT DETECTION
    # Detector 1.1 - Data Drift
    data_drift_test_suite = {'tests': [
        DataDriftTestPreset(stattest_threshold=data_stattest_threshold),
    ]}
    d1 = SimpleMonitoringStrategy(n_curr=n_curr, n_ref=n_ref)
    ev1 = EvidentlyMonitoringStep(data_drift_test_suite, d1, logger_factory, min_instances=2*n_curr, clock=n_curr, name='data_drift_eval')

    # Detector 1.2 - Target Drift
    target_drift_tests = {'tests': [
        TestColumnDrift(column_name='target', stattest_threshold=target_stattest_threshold),
    ]}
    d2 = SimpleMonitoringStrategy(n_curr=n_curr, n_ref=n_ref, type='target')
    ev2 = EvidentlyMonitoringStep(target_drift_tests, d2, logger_factory, min_instances=2*n_curr, clock=n_curr, name='target_drift_eval')

    monitoring_steps = []
    if data_drift:
        monitoring_steps.append((ev1.name, ev1))
    if target_drift:
        monitoring_steps.append((ev2.name, ev2))
    input_monitoring = DriftMonitoringPipeline(monitoring_steps)
    
    input_drift_retraining_strategy = FromLastReplacementRetrainingStrategy()#SimpleRetrainingStrategy(n_last_retrain=n_curr, n_last_test=0)#FromLastReplacementRetrainingStrategy()
    input_detector = DriftHandler(input_monitoring, input_drift_retraining_strategy)

    return input_detector
    
def get_evidently_output_handlers(n_curr, n_ref, logger_factory):
    ### OUTPUT (PERFORMANCE) DRIFT DETECTION
    # Detector 2.1 - Performance Drift

    performance_drift = {'tests': [
        TestPrecisionScore(),
        TestRecallScore(),
        TestF1Score(),
        TestAccuracyScore()
    ]}
    d3 = SimpleMonitoringStrategy(n_curr=n_curr, n_ref=n_ref, type='prediction')
    ev3 = EvidentlyMonitoringStep(performance_drift, d3, logger_factory, min_instances=2*n_curr, clock=n_curr, name='performance_drift_eval')

    output_monitoring = DriftMonitoringPipeline([(ev3.name, ev3)])
    output_drift_retraining_strategy = FromLastReplacementRetrainingStrategy()#SimpleRetrainingStrategy(n_last_retrain=n_curr, n_last_test=0)#FromLastReplacementRetrainingStrategy()
    output_detector = DriftHandler(output_monitoring, output_drift_retraining_strategy)
    
    return output_detector


def get_adwin_input_handlers(df_columns, clock=5000, grace_period=5000, min_window_length=1000, delta=1e-5, adwin_detector_type='all', logger_factory=None):
    ### INPUT DRIFT DETECTION
    # Detector 1.1 - ADWIN
    
    adwins = []
    if adwin_detector_type == 'data_only' or adwin_detector_type == 'all':

        j = 0
        for col in df_columns:
            if col == 'dataset': continue
            if col == 'target': continue
            adwin = RiverMonitoringStep(col, j, drift.ADWIN(clock=clock, grace_period=grace_period, min_window_length=min_window_length, delta=delta), logger_factory)
            adwins.append(adwin)
            j += 1

        
    if adwin_detector_type == 'target_only' or adwin_detector_type == 'all':
        ### INPUT DRIFT DETECTION
        # Detector 1.2 - ADWIN
        
        adwin = RiverMonitoringStep('target', -1, drift.ADWIN(clock=clock, grace_period=grace_period, min_window_length=min_window_length, delta=delta), logger_factory)
        adwins.append(adwin)

    input_monitoring = DriftMonitoringPipeline([(a.name, a) for a in adwins])
    input_drift_retraining_strategy = SimpleRetrainingStrategy(n_last_retrain=clock, n_last_test=0)
    input_detector = DriftHandler(input_monitoring, input_drift_retraining_strategy)

    return input_detector
    ###

def get_eval_pipeline(window_size):
    eval_pipe = RiverEvaluationPipeline(metric_steps=[
        (f'acc_preq_{window_size}', Rolling(Accuracy(), window_size)),
        (f'macro_f1_preq_{window_size}', Rolling(MacroF1(), window_size)),
        (f'kappa_preq_{window_size}', Rolling(CohenKappa(), window_size)),
        ('acc', Accuracy()),
        ('f1_macro', MacroF1()),
        ('kappa', CohenKappa())
    ])
    return eval_pipe

def get_batch_pipeline(n_curr, n_first_fit, sklearn_pipe, input_handlers, output_handlers, model_comparer, logger_factory):
    history = HistoryManager()
    batch_pipeline = BatchPipeline(
        sklearn_pipe,
        input_drift_handlers=input_handlers,
        output_drift_handlers=output_handlers,
        history=history,
        logger_factory=logger_factory,
        model_comparer=model_comparer,
        min_samples_retrain=n_curr,
        min_samples_first_fit=n_first_fit
    )
    return batch_pipeline

def get_evidently_experiment(suffix, sklearn_pipeline, n_online=500, n_first_fit=500, window_size=1000, n_curr=5_000, data_stattest_threshold=0.05, target_stattest_threshold=0.05,
    data_drift=True, target_drift=True, is_performance=True):
    prefix = str(uuid.uuid4())[:8]
    name = f'{prefix}_evidently_d_{data_drift}_t_{target_drift}__{suffix}'
    exp_name = f'{name}_{datetime.today().strftime("%Y%m%d_%H%M%S")}'
    
    logger_factory = LoggerFactory(exp_name)
    input_handlers = get_evidently_input_handlers(n_curr=n_curr, n_ref=n_curr,
        data_stattest_threshold=data_stattest_threshold, target_stattest_threshold=target_stattest_threshold,
        logger_factory=logger_factory, data_drift=data_drift, target_drift=target_drift)
    output_handlers = get_evidently_output_handlers(n_curr, n_ref=n_curr, logger_factory=logger_factory) if is_performance else None
    sklearn_pipeline = SklearnEstimator(sklearn_pipeline)
    model_comparer = ShadowOnlineComparer(n_online=n_online)
    batch_pipeline = get_batch_pipeline(n_curr, n_first_fit, sklearn_pipeline, input_handlers, output_handlers, model_comparer, logger_factory)
    eval_pipe = get_eval_pipeline(window_size)
    experiment = StreamExperiment(batch_pipeline, eval_pipe, logger_factory)
    return experiment


def get_adwin_experiment(suffix, sklearn_pipeline, n_online=500, n_first_fit=500, window_size=1000,
    clock=5000, grace_period=5000, min_window_length=1000, delta=1e-5, adwin_detector_type='all', df_columns=None):
    prefix = str(uuid.uuid4())[:8]
    name = f'{prefix}_adwin_{adwin_detector_type}_{suffix}'
    exp_name = f'{name}_{datetime.today().strftime("%Y%m%d_%H%M%S")}'
    
    logger_factory = LoggerFactory(exp_name)
    input_handlers = get_adwin_input_handlers(df_columns, clock, grace_period, min_window_length, delta, adwin_detector_type, logger_factory)
    output_handlers = None
    sklearn_pipeline = SklearnEstimator(sklearn_pipeline)
    model_comparer = ShadowOnlineComparer(n_online=n_online)
    batch_pipeline = get_batch_pipeline(clock, n_first_fit, sklearn_pipeline, input_handlers, output_handlers, model_comparer, logger_factory)
    eval_pipe = get_eval_pipeline(window_size)
    experiment = StreamExperiment(batch_pipeline, eval_pipe, logger_factory)
    return experiment

def get_online_experiment(suffix, river_model, window_size=1000):
    prefix = str(uuid.uuid4())[:8]
    model_name = str(river_model.__class__).split('.')[-1].replace("'>", "")
    name = f'{prefix}_{model_name}_{suffix}'
    exp_name = f'{name}_{datetime.today().strftime("%Y%m%d_%H%M%S")}'
    logger_factory = LoggerFactory(experiment_id=exp_name)
    eval_pipe = get_eval_pipeline(window_size)
    river_pipe = RiverPipeline(river_model)
    experiment = StreamExperiment(river_pipe, eval_pipe, logger_factory)
    return experiment

def get_combining_experiment(suffix, sklearn_estimators, river_estimators, window_size, n_curr=5_000, data_stattest_threshold=0.05, target_stattest_threshold=0.05,
    n_first_fit=5_000, n_online=500, is_data_drift=True, is_target_drift=True, is_performance=True, comb_type='mv'):
    prefix = str(uuid.uuid4())[:8]
    name = f'{prefix}_combine_{suffix}'
    exp_name = f'{name}_{datetime.today().strftime("%Y%m%d_%H%M%S")}'
    logger_factory = LoggerFactory(experiment_id=exp_name)
    eval_pipe = get_eval_pipeline(window_size)
    
    members = []
    for sklearn_estimator in sklearn_estimators:
        members.append(_construct_batch_member(n_curr, n_first_fit, data_stattest_threshold, target_stattest_threshold, sklearn_estimator,
            n_online, logger_factory, is_data_drift, is_target_drift, is_performance))
    for river_estimator in river_estimators:
        members.append(RiverPipeline(river_estimator))

    if comb_type == 'mv': combiner = MajorityVoteCombiner()
    elif comb_type == 'ds': combiner = DynamicSwitchCombiner(n_members=len(members), metric=Rolling(MacroF1(), window_size), logger_factory=logger_factory)
    elif comb_type == 'wv': combiner = WeightedVoteCombiner(n_members=len(members), metric=Rolling(MacroF1(), window_size), logger_factory=logger_factory)
    else: raise ValueError('comb_type not recognized')
    comb_pipeline = CombinationPipeline(members=members, combiner=combiner)
    
    experiment = StreamExperiment(comb_pipeline, eval_pipe, logger_factory)
    return experiment

def _construct_batch_member(n_curr, n_first_fit, data_stattest_threshold, target_stattest_threshold, sklearn_estimator,
        n_online, logger_factory, is_data_drift, is_target_drift, is_performance):
    input_handlers = get_evidently_input_handlers(n_curr=n_curr, n_ref=n_curr,
        data_stattest_threshold=data_stattest_threshold, target_stattest_threshold=target_stattest_threshold,
        logger_factory=logger_factory, data_drift=is_data_drift, target_drift=is_target_drift)
    output_handlers = get_evidently_output_handlers(n_curr, n_ref=n_curr, logger_factory=logger_factory) if is_performance else None
    model_comparer = ShadowOnlineComparer(n_online=n_online)
    sklearn_pipeline = SklearnEstimator(sklearn_estimator)
    batch_pipeline = get_batch_pipeline(n_curr, n_first_fit, sklearn_pipeline, input_handlers, output_handlers, model_comparer, logger_factory)
    return batch_pipeline

def get_combining_exp_different_detectors(suffix, sklearn_estimators, river_estimators, window_size, n_curr, n_first_fit, n_online,
    data_stattest_threshold, target_stattest_threshold, is_performance, comb_type):
    prefix = str(uuid.uuid4())[:8]
    name = f'{prefix}_combine_{suffix}'
    exp_name = f'{name}_{datetime.today().strftime("%Y%m%d_%H%M%S")}'
    logger_factory = LoggerFactory(experiment_id=exp_name)
    eval_pipe = get_eval_pipeline(window_size)

    members = []
    for sklearn_estimator in sklearn_estimators:
        if data_stattest_threshold != None:
            if type(data_stattest_threshold) is not list:
                data_stattest_threshold = [data_stattest_threshold]
            for td in data_stattest_threshold:
                members.append(_construct_batch_member(n_curr, n_first_fit, td, 0.05, sklearn_estimator,
                    n_online, logger_factory, True, False, False))
                
        if target_stattest_threshold != None:
            if type(target_stattest_threshold) is not list:
                data_stattest_threshold = [target_stattest_threshold]
            for tt in target_stattest_threshold:
                members.append(_construct_batch_member(n_curr, n_first_fit, 0.05, tt, sklearn_estimator,
                    n_online, logger_factory, False, True, False))
                
        if is_performance:
            members.append(_construct_batch_member(n_curr, n_first_fit, 0.05, 0.05, sklearn_estimator,
                n_online, logger_factory, False, False, True))
            
    for river_estimator in river_estimators:
        members.append(RiverPipeline(river_estimator))

    if comb_type == 'mv': combiner = MajorityVoteCombiner()
    elif comb_type == 'ds': combiner = DynamicSwitchCombiner(n_members=len(members), metric=Rolling(MacroF1(), window_size), logger_factory=logger_factory)
    elif comb_type == 'wv': combiner = WeightedVoteCombiner(n_members=len(members), metric=Rolling(MacroF1(), window_size), logger_factory=logger_factory)
    else: raise ValueError('comb_type not recognized')
    comb_pipeline = CombinationPipeline(members=members, combiner=combiner)
    
    experiment = StreamExperiment(comb_pipeline, eval_pipe, logger_factory)
    return experiment

def get_combining_exp_different_s(suffix, sklearn_estimators, river_estimators, window_size, n_curr_s, n_first_fit, n_online,
    data_stattest_threshold, target_stattest_threshold, is_performance, comb_type):
    prefix = str(uuid.uuid4())[:8]
    name = f'{prefix}_combine_{suffix}'
    exp_name = f'{name}_{datetime.today().strftime("%Y%m%d_%H%M%S")}'
    logger_factory = LoggerFactory(experiment_id=exp_name)
    eval_pipe = get_eval_pipeline(window_size)

    is_data_detection = data_stattest_threshold != None
    is_target_detection = target_stattest_threshold != None
    members = []
    for sklearn_estimator in sklearn_estimators:
        if n_curr_s != None:
            if type(n_curr_s) is not list:
                n_curr_s = [n_curr_s]
            for n_c in n_curr_s:
                members.append(_construct_batch_member(n_c, n_first_fit, data_stattest_threshold, target_stattest_threshold, sklearn_estimator,
                    n_online, logger_factory, is_data_detection, is_target_detection, is_performance))
                
    for river_estimator in river_estimators:
        members.append(RiverPipeline(river_estimator))

    if comb_type == 'mv': combiner = MajorityVoteCombiner()
    elif comb_type == 'ds': combiner = DynamicSwitchCombiner(n_members=len(members), metric=Rolling(MacroF1(), window_size), logger_factory=logger_factory)
    elif comb_type == 'wv': combiner = WeightedVoteCombiner(n_members=len(members), metric=Rolling(MacroF1(), window_size), logger_factory=logger_factory)
    else: raise ValueError('comb_type not recognized')
    comb_pipeline = CombinationPipeline(members=members, combiner=combiner)
    
    experiment = StreamExperiment(comb_pipeline, eval_pipe, logger_factory)
    return experiment
 