from river.forest import ARFClassifier
from batchstream.utils.logging.base.logger_factory import LoggerFactory
from batchstream.pipelines.stream.model_river_pipeline import RiverPipeline
from batchstream.experiment.experiment import StreamExperiment
from river.metrics import Accuracy, MacroF1, CohenKappa
from river.utils import Rolling
from batchstream.evaluation.river_evaluation_pipeline import RiverEvaluationPipeline
from datetime import datetime
import uuid
from river import ensemble
from river import tree
from river.tree import HoeffdingAdaptiveTreeClassifier
from river import naive_bayes



def get_arf_exp(seed=42, window_size=1000, suffix=''):
    prefix = str(uuid.uuid4())[:8]
    name = f'{prefix}_arf_{suffix}'
    exp_name = f'{name}_{datetime.today().strftime("%Y%m%d_%H%M%S")}'
    logger_factory = LoggerFactory(experiment_id=exp_name)

    eval_pipe = RiverEvaluationPipeline(metric_steps=[
        (f'acc_preq_{window_size}', Rolling(Accuracy(), window_size)),
        (f'macro_f1_preq_{window_size}', Rolling(MacroF1(), window_size)),
        (f'kappa_preq_{window_size}', Rolling(CohenKappa(), window_size)),
        ('acc', Accuracy()),
        ('f1_macro', MacroF1()),
        ('kappa', CohenKappa())
    ])

    arf_model = ARFClassifier(seed=seed, leaf_prediction="mc", drift_detector=A)
    arf_pipe = RiverPipeline(arf_model)
    arf_experiment = StreamExperiment(arf_pipe, eval_pipe, logger_factory)
    return arf_experiment

def get_srp_exp(seed=42, window_size=1000, nominal_attributes=None, suffix=''):
    prefix = str(uuid.uuid4())[:8]
    name = f'{prefix}_srp_{suffix}'
    exp_name = f'{name}_{datetime.today().strftime("%Y%m%d_%H%M%S")}'
    logger_factory = LoggerFactory(experiment_id=exp_name)

    eval_pipe = RiverEvaluationPipeline(metric_steps=[
        (f'acc_preq_{window_size}', Rolling(Accuracy(), window_size)),
        (f'macro_f1_preq_{window_size}', Rolling(MacroF1(), window_size)),
        (f'kappa_preq_{window_size}', Rolling(CohenKappa(), window_size)),
        ('acc', Accuracy()),
        ('f1_macro', MacroF1()),
        ('kappa', CohenKappa())
    ])


    base_model = tree.HoeffdingTreeClassifier(grace_period=50, delta=0.01, nominal_attributes=nominal_attributes)
    srp_model = ensemble.SRPClassifier(model=base_model, n_models=3, seed=seed)
    srp_pipe = RiverPipeline(srp_model)
    srp_experiment = StreamExperiment(srp_pipe, eval_pipe, logger_factory)
    return srp_experiment


def get_hoeffding_tree_exp(seed=42, window_size=1000, suffix=''):
    prefix = str(uuid.uuid4())[:8]
    name = f'{prefix}_ht_{suffix}'
    exp_name = f'{name}_{datetime.today().strftime("%Y%m%d_%H%M%S")}'
    logger_factory = LoggerFactory(experiment_id=exp_name)

    eval_pipe = RiverEvaluationPipeline(metric_steps=[
        (f'acc_preq_{window_size}', Rolling(Accuracy(), window_size)),
        (f'macro_f1_preq_{window_size}', Rolling(MacroF1(), window_size)),
        (f'kappa_preq_{window_size}', Rolling(CohenKappa(), window_size)),
        ('acc', Accuracy()),
        ('f1_macro', MacroF1()),
        ('kappa', CohenKappa())
    ])

    hat_model = HoeffdingAdaptiveTreeClassifier(
        grace_period=100,
        delta=1e-5,
        leaf_prediction='nb',
        nb_threshold=10,
        seed=seed
        )
    hat_pipe = RiverPipeline(hat_model)
    hat_experiment = StreamExperiment(hat_pipe, eval_pipe, logger_factory)
    return hat_experiment

def get_naive_bayes_online_exp(seed=42, window_size=1000, suffix=''):
    prefix = str(uuid.uuid4())[:8]
    name = f'{prefix}_nb_{suffix}'
    exp_name = f'{name}_{datetime.today().strftime("%Y%m%d_%H%M%S")}'
    logger_factory = LoggerFactory(experiment_id=exp_name)

    eval_pipe = RiverEvaluationPipeline(metric_steps=[
        (f'acc_preq_{window_size}', Rolling(Accuracy(), window_size)),
        (f'macro_f1_preq_{window_size}', Rolling(MacroF1(), window_size)),
        (f'kappa_preq_{window_size}', Rolling(CohenKappa(), window_size)),
        ('acc', Accuracy()),
        ('f1_macro', MacroF1()),
        ('kappa', CohenKappa())
    ])

    
    nb_model =  naive_bayes.GaussianNB()
    nb_pipe = RiverPipeline(nb_model)    
    nb_experiment = StreamExperiment(nb_pipe, eval_pipe, logger_factory)
    return nb_experiment
