from batchstream.utils.concurrent import run_concurrent
from experiments.first.rf_adwin_only import get_rf_adwin_only_exp
from experiments.first.rf_adwin_target import get_rf_adwin_target_exp
from experiments.first.rf_data_evidently import get_rf_data_evidently_exp
from experiments.first.rf_target_evidently import get_rf_target_evidently_exp
from experiments.first.rf_perf_evidently import get_rf_perf_evidently_exp
from utils.read_data.get_dataset import get_dataset
import uuid



NUM_WORKERS = 20

def compose_experiments(dataset_name):


    suffix = f"{str(uuid.uuid4())[:4]}_{dataset_name}"
    df = get_dataset(dataset_name)
    args_list = []
    
    window_size = 100

    clock=50
    grace_period=100
    min_window_length=20
    n_first_fit=300
    args_list.append((get_rf_adwin_target_exp(df, window_size=window_size, suffix=suffix, clock=clock, grace_period=grace_period, min_window_length=min_window_length, n_first_fit=n_first_fit), df)) # rf + adwin (data + target)
    
    clock=100
    grace_period=100
    min_window_length=50
    n_first_fit=1000
    args_list.append((get_rf_adwin_target_exp(df, window_size=window_size, suffix=suffix, clock=clock, grace_period=grace_period, min_window_length=min_window_length, n_first_fit=n_first_fit), df)) # rf + adwin (data + target)

    clock=20
    grace_period=100
    min_window_length=30
    n_first_fit=300
    args_list.append((get_rf_adwin_target_exp(df, window_size=window_size, suffix=suffix, clock=clock, grace_period=grace_period, min_window_length=min_window_length, n_first_fit=n_first_fit), df)) # rf + adwin (data + target)
    
    
    n_curr = 1000
    n_ref = 1000
    n_first_fit=1000
    args_list.append((get_rf_target_evidently_exp(suffix=suffix, window_size=window_size, n_curr=n_curr, n_ref=n_ref, n_first_fit=n_first_fit), df)) # rf + target evidently
    args_list.append((get_rf_data_evidently_exp(suffix=suffix, window_size=window_size, n_ref=n_ref, n_first_fit=n_first_fit), df))
    args_list.append((get_rf_perf_evidently_exp(suffix=suffix, window_size=window_size, n_ref=n_ref, n_first_fit=n_first_fit), df))

    n_curr = 2_500
    n_ref = 2_500
    n_first_fit=1_000
    args_list.append((get_rf_target_evidently_exp(suffix=suffix, window_size=window_size, n_ref=n_ref, n_first_fit=n_first_fit), df)) # rf + target evidently
    args_list.append((get_rf_data_evidently_exp(suffix=suffix, window_size=window_size, n_ref=n_ref, n_first_fit=n_first_fit), df))
    args_list.append((get_rf_perf_evidently_exp(suffix=suffix, window_size=window_size, n_ref=n_ref, n_first_fit=n_first_fit), df))

    n_curr = 500
    n_ref = 500
    n_first_fit=500
    args_list.append((get_rf_target_evidently_exp(suffix=suffix, window_size=window_size, n_ref=n_ref, n_first_fit=n_first_fit), df)) # rf + target evidently
    args_list.append((get_rf_data_evidently_exp(suffix=suffix, window_size=window_size, n_ref=n_ref, n_first_fit=n_first_fit), df))
    args_list.append((get_rf_perf_evidently_exp(suffix=suffix, window_size=window_size, n_ref=n_ref, n_first_fit=n_first_fit), df))
    
    n_curr = 5000
    n_ref = 5000
    n_first_fit=2500
    args_list.append((get_rf_target_evidently_exp(suffix=suffix, window_size=window_size, n_ref=n_ref, n_first_fit=n_first_fit), df)) # rf + target evidently
    args_list.append((get_rf_data_evidently_exp(suffix=suffix, window_size=window_size, n_ref=n_ref, n_first_fit=n_first_fit), df))
    args_list.append((get_rf_perf_evidently_exp(suffix=suffix, window_size=window_size, n_ref=n_ref, n_first_fit=n_first_fit), df))

    return args_list

def main():
    args_list = compose_experiments('elec')
    run_concurrent(args_list, NUM_WORKERS)
    
if __name__ == "__main__":
    main()
