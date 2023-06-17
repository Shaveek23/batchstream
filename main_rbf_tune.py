from batchstream.utils.concurrent import run_concurrent
from experiments.second import get_adwin_experiment, get_evidently_experiment
from utils.read_data.get_dataset import get_dataset
import uuid
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import xgboost as xgb




NUM_WORKERS = 54

def compose_evidently_experiments(dataset_name):
    suffix = f"{str(uuid.uuid4())[:4]}_{dataset_name}"
    df = get_dataset(dataset_name)
    args_list = []

    # COMMON HYPERPARAMETERS
    window_size = 1000
    n_first_fit = 5000
    rf = Pipeline([('rf', RandomForestClassifier(random_state=42))])

    param_grid = {
        'n_curr': [1000, 5000, 2500, 10_000],
        'stattest_threshold': [0.02, 0.03, 0.035]
    }
    samples = list(ParameterSampler(param_grid, n_iter=12, random_state=42))

    for samples in samples:
        n_curr = samples['n_curr']
        n_ref = n_curr
        stattest_threshold = samples['stattest_threshold']
        args_list.append((get_evidently_experiment(suffix, rf, n_online=500, n_first_fit=n_first_fit, window_size=window_size,
        n_curr=n_curr, data_stattest_threshold=stattest_threshold, target_stattest_threshold=stattest_threshold,
        data_drift=True, target_drift=True, is_performance=True), df.copy(deep=True)))
        
        args_list.append((get_evidently_experiment(suffix, rf, n_online=500, n_first_fit=n_first_fit, window_size=window_size,
        n_curr=n_curr, data_stattest_threshold=stattest_threshold, target_stattest_threshold=stattest_threshold,
        data_drift=True, target_drift=False, is_performance=False), df.copy(deep=True)))
        
        args_list.append((get_evidently_experiment(suffix, rf, n_online=500, n_first_fit=n_first_fit, window_size=window_size,
        n_curr=n_curr, data_stattest_threshold=stattest_threshold, target_stattest_threshold=stattest_threshold,
        data_drift=False, target_drift=True, is_performance=False), df.copy(deep=True)))

    args_list.append((get_evidently_experiment(suffix, rf, n_online=500, n_first_fit=n_first_fit, window_size=window_size,
    n_curr=n_curr, data_stattest_threshold=stattest_threshold, target_stattest_threshold=stattest_threshold,
    data_drift=False, target_drift=False, is_performance=True), df.copy(deep=True)))
        
    return args_list

def compose_adwin_experiments(dataset_name):

    suffix = f"{str(uuid.uuid4())[:4]}_{dataset_name}"
    df = get_dataset(dataset_name)
    args_list = []


    window_size = 1000
    grace_period = 5000
    n_first_fit = 5000
    rf = Pipeline([('rf', RandomForestClassifier(random_state=42))])


    param_grid = {
        'clock': [100, 5000, 2500, 50],
        'delta': [0.05, 0.1, 0.2, 0.5],
        'min_window_length': [50, 10, 500]
    }
    samples = list(ParameterSampler(param_grid, n_iter=12, random_state=42))

    for sample in samples:
        args_list.append((
                get_adwin_experiment(suffix, rf, window_size=window_size, df_columns=df.columns, grace_period=grace_period, n_first_fit=n_first_fit, **sample),
                df.copy(deep=True)
            )) 
        args_list.append((
            get_adwin_experiment(suffix, rf, window_size=window_size, adwin_detector_type='target_only', df_columns=df.columns, grace_period=grace_period, n_first_fit=n_first_fit, **sample),
            df.copy(deep=True)
        )) 
        args_list.append((
            get_adwin_experiment(suffix, rf, window_size=window_size, adwin_detector_type='data_only', df_columns=df.columns, grace_period=grace_period, n_first_fit=n_first_fit, **sample),
            df.copy(deep=True)
        )) 
    return args_list

def main():
    dataset_name = 'rbf66'
    args_list = []
    args_list += compose_evidently_experiments(dataset_name)
    args_list += compose_adwin_experiments(dataset_name)
    run_concurrent(args_list, NUM_WORKERS)
    
if __name__ == "__main__":
    main()
