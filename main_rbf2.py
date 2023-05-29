from batchstream.utils.concurrent import run_concurrent
from experiments.second import get_evidently_experiment, get_online_experiment
from utils.read_data.get_dataset import get_dataset
import uuid
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
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


NUM_WORKERS = 8

def compose_experiments(dataset_name):
    suffix_ = f"{str(uuid.uuid4())[:4]}_{dataset_name}"
    df = get_dataset(dataset_name)
    args_list = []

    # COMMON HYPERPARAMETERS
    window_size = 1000
    n_first_fit = 5000
    seed = 42
    

    ###
    # Batch methods:
    suffix = f"batch_{suffix_}"
    n_curr_ref_retrain = 10_000
    threshold = 0.03
    n_online = 500

    # LOGISTIC REGRESSION
    # lr = Pipeline([('sc', SC()), ('lr_batch', LogisticRegression())])
    # args_list.append((get_evidently_experiment(f"{suffix}_lr", lr, n_online=n_online, n_first_fit=n_first_fit, window_size=window_size,
    #     n_curr=n_curr_ref_retrain, data_stattest_threshold=threshold,
    #     data_drift=True, target_drift=False, is_performance=False), df.copy(deep=True)))
    
    # DECISION TREE CLASSIFIER
    dt = Pipeline([('dt', DecisionTreeClassifier())])
    # args_list.append((get_evidently_experiment(f"{suffix}_dt", dt, n_online=n_online, n_first_fit=n_first_fit, window_size=window_size,
    #     n_curr=n_curr_ref_retrain, data_stattest_threshold=threshold,
    #     data_drift=True, target_drift=False, is_performance=False), df.copy(deep=True)))
    
    # # NAIVE BAYES
    # nb = Pipeline([('nb_batch', GaussianNB())])
    # args_list.append((get_evidently_experiment(f"{suffix}_nb", nb, n_online=n_online, n_first_fit=n_first_fit, window_size=window_size,
    #     n_curr=n_curr_ref_retrain, data_stattest_threshold=threshold,
    #     data_drift=True, target_drift=False, is_performance=False), df.copy(deep=True)))
    
    # # XGBoost 
    xgb_batch = Pipeline([('xgb', xgb.XGBClassifier())])
    # args_list.append((get_evidently_experiment(f"{suffix}_xgb", xgb_batch, n_online=n_online, n_first_fit=n_first_fit, window_size=window_size,
    #     n_curr=n_curr_ref_retrain, data_stattest_threshold=threshold,
    #     data_drift=True, target_drift=False, is_performance=False), df.copy(deep=True)))
    
    ##
    rf = Pipeline([('rf', RandomForestClassifier(random_state=42))])
    ###
    # # Online methods
    suffix = f"online_{suffix_}"

    # # ADAPTIVE RANDOM FOREST
    arf = forest.ARFClassifier(seed=seed, leaf_prediction="mc")
    #args_list.append((get_online_experiment(suffix, arf, window_size),  df.copy(deep=True)))

    # LOGISTIC REGRESSION
    lr_online = compose.Pipeline(
        StandardScaler(
            with_std=True
        ),
        LROnline(
            optimizer=optim.SGD(
                lr=0.005
            ),
            loss=optim.losses.Log(
                weight_pos=1.,
                weight_neg=1.
            ),
            l2=1.0,
            l1=0.,
            intercept_init=0.,
            intercept_lr=0.01,
            clip_gradient=1e+12,
            initializer=optim.initializers.Zeros()
        )
    )
    # args_list.append((get_online_experiment(suffix, lr_online, window_size),  df.copy(deep=True)))

    # # HOEFFDING TREE 
    hat = HoeffdingAdaptiveTreeClassifier(
        grace_period=100,
        delta=0.01,
        leaf_prediction='nb',
        nb_threshold=10,
        seed=seed
    )
    # args_list.append((get_online_experiment(suffix, hat.clone(), window_size),  df.copy(deep=True)))

    # # NAIVE BAYES
    # nb_online = GNBOnline()
    # args_list.append((get_online_experiment(suffix, nb_online, window_size),  df.copy(deep=True)))
   
    # # SRP
    nominal_attributes = []
    base_model = HoeffdingTreeClassifier(grace_period=100, delta=0.01, nominal_attributes=nominal_attributes)
    srp_model = ensemble.SRPClassifier(model=base_model, n_models=3, seed=seed)
    # args_list.append((get_online_experiment(suffix, srp_model, window_size),  df.copy(deep=True)))


    ## # COMBINING BATCH AND ONLINE
    suffix = f"combining_{suffix_}"

    args_list.append((get_combining_experiment(suffix, [copy.deepcopy(xgb_batch), copy.deepcopy(dt)], [srp_model.clone()], window_size, n_curr_ref_retrain, threshold,
        threshold, n_first_fit, n_online, is_data_drift=True, is_target_drift=False, is_performance=False, comb_type='mv'), df.copy(deep=True)))
    
    # args_list.append((get_combining_experiment(suffix, [copy.deepcopy(rf)], [hat.clone()], window_size, n_curr_ref_retrain, threshold,
    #     threshold, n_first_fit, n_online, is_data_drift=True, is_target_drift=False, is_performance=False, comb_type='ds'), df.copy(deep=True)))
    
    args_list.append((get_combining_experiment(suffix, [copy.deepcopy(rf)], [arf.clone(), hat.clone()], window_size, n_curr_ref_retrain, threshold,
        threshold, n_first_fit, n_online, is_data_drift=True, is_target_drift=False, is_performance=False, comb_type='mv'), df.copy(deep=True)))
    
    # args_list.append((get_combining_experiment(suffix, [copy.deepcopy(xgb_batch)], [arf.clone()], window_size, n_curr_ref_retrain, threshold,
    #     threshold, n_first_fit, n_online, is_data_drift=True, is_target_drift=False, is_performance=False, comb_type='ds'), df.copy(deep=True)))

    return args_list

def main():
    dataset_name = 'rbf66'
    args_list = []
    args_list += compose_experiments(dataset_name)
    #run_concurrent(args_list, NUM_WORKERS)
    len(args_list)
    
if __name__ == "__main__":
    main()
