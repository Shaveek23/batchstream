from batchstream.utils.concurrent import run_concurrent
from experiments.first.rf_adwin_only import get_rf_adwin_only_exp
from experiments.first.rf_adwin_target import get_rf_adwin_target_exp
from experiments.first.rf_data_evidently import get_rf_data_evidently_exp
from experiments.first.rf_target_evidently import get_rf_target_evidently_exp
from experiments.first.rf_perf_evidently import get_rf_perf_evidently_exp
from experiments.first.rf_all_evidently import get_rf_all_evidently_exp
from utils.read_data.get_dataset import get_dataset
import uuid
from sklearn.model_selection import ParameterSampler



NUM_WORKERS = 54

def compose_evidently_experiments(dataset_name):
    suffix = f"{str(uuid.uuid4())[:4]}_{dataset_name}"
    df = get_dataset(dataset_name)
    args_list = []

    # COMMON HYPERPARAMETERS
    window_size = 1000
    n_first_fit = 500

    param_grid = {
        'n_curr': [1000, 5000, 2500, 10_000],
        'stattest_threshold': [0.038, 0.04, 0.043]
    }
    samples = list(ParameterSampler(param_grid, n_iter=10, random_state=42))

    for samples in samples:
        n_curr = samples['n_curr']
        n_ref = n_curr
        stattest_threshold = samples['stattest_threshold']
        args_list.append((get_rf_target_evidently_exp(suffix=suffix, window_size=window_size, n_ref=n_ref, n_first_fit=n_first_fit,
                                                    stattest_threshold=stattest_threshold), df))
        args_list.append((get_rf_data_evidently_exp(suffix=suffix, window_size=window_size, n_ref=n_ref, n_first_fit=n_first_fit, 
                                                    stattest_threshold=stattest_threshold), df))
        args_list.append((get_rf_perf_evidently_exp(suffix=suffix, window_size=window_size, n_ref=n_ref, n_first_fit=n_first_fit), df))
        args_list.append((get_rf_all_evidently_exp(suffix=suffix, window_size=window_size, n_ref=n_ref, n_first_fit=n_first_fit, 
                                                   data_stattest_threshold=stattest_threshold, target_stattest_threshold=stattest_threshold)))
    return args_list

def compose_adwin_experiments(dataset_name):

    suffix = f"{str(uuid.uuid4())[:4]}_{dataset_name}"
    df = get_dataset(dataset_name)
    args_list = []


    window_size = 1000
    grace_period = 1000
    n_first_fit = 300


    param_grid = {
        'clock': [100, 5000, 2500, 50],
        'delta': [0.02, 0.01, 0.2],
        'min_window_length': [50]
    }
    samples = list(ParameterSampler(param_grid, n_iter=10, random_state=42))

    for sample in samples:
        args_list.append((get_rf_adwin_target_exp(df, window_size=window_size, suffix=suffix, grace_period=grace_period, n_first_fit=n_first_fit, **sample), df)) # rf + adwin (target)
        args_list.append((get_rf_adwin_only_exp(df, window_size=window_size, suffix=suffix, grace_period=grace_period, n_first_fit=n_first_fit, **sample), df)) # rf + adwin (data)
    return args_list

def main():
    args_list = compose_evidently_experiments('rbf66')
    run_concurrent(args_list, NUM_WORKERS)
    
if __name__ == "__main__":
    main()
