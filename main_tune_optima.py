from batchstream.utils.concurrent import run_concurrent
from experiments.second import get_adwin_experiment, get_evidently_experiment
from utils.read_data.get_dataset import get_dataset
import uuid
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import copy



NUM_WORKERS = 48

def compose_evidently_experiments(dataset_name):
    suffix = f"{str(uuid.uuid4())[:4]}_{dataset_name}"
    df = get_dataset(dataset_name)
    args_list = []

    # COMMON HYPERPARAMETERS
    window_size = 100
    n_online = 50
    n_first_fit = 150
    rf = Pipeline([('rf', RandomForestClassifier())])

    param_grid = {
        'n_curr': [50, 100, 250, 500],
        'stattest_threshold': [0.01, 0.02, 0.03]
    }
    samples = list(ParameterSampler(param_grid, n_iter=12, random_state=42))

    for samples in samples:
        n_curr_ref_retrain = samples['n_curr']
        threshold = samples['stattest_threshold']
        
        args_list.append((get_evidently_experiment(f"{suffix}_rf", copy.deepcopy(rf), n_online=n_online, n_first_fit=n_first_fit, window_size=window_size,
            n_curr=n_curr_ref_retrain, data_stattest_threshold=threshold,
            data_drift=True, target_drift=False, is_performance=False), df.copy(deep=True)))
        
        args_list.append((get_evidently_experiment(f"{suffix}_rf", copy.deepcopy(rf), n_online=n_online, n_first_fit=n_first_fit, window_size=window_size,
            n_curr=n_curr_ref_retrain, data_stattest_threshold=threshold,
            data_drift=True, target_drift=True, is_performance=True), df.copy(deep=True)))
        
        args_list.append((get_evidently_experiment(f"{suffix}_rf", copy.deepcopy(rf), n_online=n_online, n_first_fit=n_first_fit, window_size=window_size,
            n_curr=n_curr_ref_retrain, data_stattest_threshold=threshold,
            data_drift=False, target_drift=True, is_performance=False), df.copy(deep=True)))
               
    return args_list

def main():
    dataset_name = 'optima'
    args_list = []
    args_list += compose_evidently_experiments(dataset_name)
    run_concurrent(args_list, NUM_WORKERS)
    
if __name__ == "__main__":
    main()
