from datetime import datetime
from batchstream.combine.diverse_vote_combiner import DiverseVoteCombiner
from batchstream.combine.dynamic_switch_combiner import DynamicSwitchCombiner
from batchstream.combine.majority_vote_combiner import MajorityVoteCombiner
from batchstream.combine.similarity_grouping_combiner import SimilarityGroupingCombiner
from batchstream.combine.weighted_vote_combiner import WeightedVoteCombiner
from batchstream.experiment.experiment import StreamExperiment
from batchstream.pipelines.combine.combination_pipeline import CombinationPipeline
from batchstream.pipelines.stream.model_river_pipeline import RiverPipeline
from batchstream.utils.concurrent import run_concurrent
from batchstream.utils.logging.base.logger_factory import LoggerFactory
from experiments.second import _construct_batch_member, get_eval_pipeline, get_evidently_experiment, get_online_experiment
from utils.read_data.get_dataset import get_dataset
import uuid
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from river.naive_bayes import GaussianNB as GNBOnline
from river.linear_model import LogisticRegression as LROnline
from river.tree import HoeffdingAdaptiveTreeClassifier, HoeffdingTreeClassifier
from river import ensemble
from river.preprocessing import StandardScaler
from river import optim
from river import compose
from river import forest
from sklearn.preprocessing import StandardScaler as SC
from experiments.second import get_combining_experiment
import copy
from lightgbm import LGBMClassifier
from river import dummy
from river.metrics import Accuracy, CohenKappa, MacroF1
from river.utils import Rolling


NUM_WORKERS = 4


def compose_experiments(dataset_name):

    comb_types = ['ds', 'wv', 'dv', 'sg']
    x = f'{str(uuid.uuid4())[:4]}'
    for comb_type in comb_types:
        suffix_ = f"{x}_{dataset_name}_{str(uuid.uuid4())[:4]}"
        df = get_dataset(dataset_name)
        args_list = []

        # COMMON HYPERPARAMETERS
        window_size = 100
        n_online = 50
        n_first_fit = 150
        seed = 42

        name = f"comb_{comb_type}_{suffix_}_"
        exp_name = f'{name}_{datetime.today().strftime("%Y%m%d_%H%M%S")}'
        
        # Batch methods:
        strategies = [
            { 's': 250, 'th': 0.02, 'is_p': True, 'is_t': True, 'is_d': True, 'last_repl': True },
            { 's': 250, 'th': 0.02, 'is_p': True, 'is_t': False, 'is_d': False, 'last_repl': False },
            { 's': 100, 'th': 0.03, 'is_p': True, 'is_t': True, 'is_d': True, 'last_repl': True },
            { 's': 1000, 'th': 0.02, 'is_p': True, 'is_t': True, 'is_d': True, 'last_repl': False }

        ]

        members = []
        i = 0
        for strategy in strategies:
            model = Pipeline([('lgbm', LGBMClassifier(random_state=42))])
            logger_factory = LoggerFactory(experiment_id=f"{exp_name}_{i}")
            member = _construct_batch_member(strategy['s'], n_first_fit, strategy['th'], strategy['th'], model,
                n_online, logger_factory, strategy['is_d'], strategy['is_t'], strategy['is_p'], strategy['last_repl'])
            members.append(member)
            i += 1


        ## Online members
        arf = forest.ARFClassifier(seed=seed, leaf_prediction="mc")

        members.append(RiverPipeline(arf))

        # # HOEFFDING TREE 
        hat = HoeffdingAdaptiveTreeClassifier(
            grace_period=100,
            delta=0.01,
            leaf_prediction='nb',
            nb_threshold=10,
            seed=seed
        )
        members.append(RiverPipeline(hat))

        nb_online = GNBOnline()
        members.append(RiverPipeline(nb_online))
        logger_factory = LoggerFactory(experiment_id=f"{exp_name}")

        if comb_type == 'mv': combiner = MajorityVoteCombiner()
        elif comb_type == 'ds': combiner = DynamicSwitchCombiner(n_members=len(members), metric=Rolling(MacroF1(), window_size), logger_factory=logger_factory)
        elif comb_type == 'wv': combiner = WeightedVoteCombiner(n_members=len(members), metric=Rolling(MacroF1(), window_size), logger_factory=logger_factory)
        elif comb_type == 'dv': combiner = DiverseVoteCombiner(n_members=len(members), K=(len(members) // 2), clock=window_size * 2, th=0.5, metric=Rolling(MacroF1(), window_size), logger_factory=logger_factory)
        elif comb_type == 'sg': combiner = SimilarityGroupingCombiner(n_members=len(members), n_wait=250, similarity_threshold=0.15, similarity_penalty=0.5, metric=MacroF1(), logger_factory=logger_factory)
        else: raise ValueError('comb_type not recognized')
        comb_pipeline = CombinationPipeline(members=members, combiner=combiner)
        
        eval_pipe = get_eval_pipeline(window_size)
       
        experiment = StreamExperiment(comb_pipeline, eval_pipe, logger_factory)

    
        args_list.append((experiment, df.copy(deep=True)))
    return args_list

def main():
    dataset_name = 'optima'
    args_list = []
    args_list += compose_experiments(dataset_name)
    run_concurrent(args_list, NUM_WORKERS)

if __name__ == "__main__":
    main()
