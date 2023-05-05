from batchstream.utils.concurrent import run_concurrent
from experiments.covtype.covtype_rf_adwin_only import get_covtype_rf_adwin_only_exp
from experiments.covtype.covtype_rf_adwin_target import get_covtype_rf_adwin_target_exp
from experiments.covtype.covtype_rf_data_evidently import get_rf_data_evidently_exp
from experiments.covtype.covtype_rf_target_evidently import get_rf_target_evidently_exp
from experiments.covtype.covtype_rf_perf_evidently import get_rf_perf_evidently_exp
from utils.read_data.covtype import get_covtype_dataset
import uuid



NUM_WORKERS = 10

def compose_experiments():


    suffix = str(uuid.uuid4())[:4]
    df = get_covtype_dataset()
    args_list = []
    # args_list.append((get_covtype_rf_adwin_only_exp(df, suffix=suffix), df)) # rf + adwin (data + target)
    # args_list.append((get_rf_target_evidently_exp(suffix=suffix), df)) # rf + target evidently
    # args_list.append((get_rf_data_evidently_exp(suffix=suffix), df))
    # args_list.append((get_rf_perf_evidently_exp(suffix=suffix), df))

    clock=500
    grace_period=1000
    min_window_length=200
    n_first_fit=1000
    args_list.append((get_covtype_rf_adwin_target_exp(df, suffix=suffix, clock=clock, grace_period=grace_period, min_window_length=min_window_length), df)) # rf + adwin (data + target)
    
    clock=1_000
    grace_period=1000
    min_window_length=200
    n_first_fit=1000
    args_list.append((get_covtype_rf_adwin_target_exp(df, suffix=suffix, clock=clock, grace_period=grace_period, min_window_length=min_window_length), df)) # rf + adwin (data + target)

    clock=2_000
    grace_period=1000
    min_window_length=300
    n_first_fit=1000
    args_list.append((get_covtype_rf_adwin_target_exp(df, suffix=suffix, clock=clock, grace_period=grace_period, min_window_length=min_window_length), df)) # rf + adwin (data + target)
    
    
    # n_curr = 10_000
    # n_ref = 10_000
    # n_first_fit=2_000
    # args_list.append((get_rf_target_evidently_exp(suffix=suffix, n_curr=n_curr, n_ref=n_ref, n_first_fit=n_first_fit), df)) # rf + target evidently
    # args_list.append((get_rf_data_evidently_exp(suffix=suffix, n_ref=n_ref, n_first_fit=n_first_fit), df))
    # args_list.append((get_rf_perf_evidently_exp(suffix=suffix, n_ref=n_ref, n_first_fit=n_first_fit), df))

    # n_curr = 2_500
    # n_ref = 2_500
    # n_first_fit=1_000
    # args_list.append((get_rf_target_evidently_exp(suffix=suffix, n_ref=n_ref, n_first_fit=n_first_fit), df)) # rf + target evidently
    # args_list.append((get_rf_data_evidently_exp(suffix=suffix, n_ref=n_ref, n_first_fit=n_first_fit), df))
    # args_list.append((get_rf_perf_evidently_exp(suffix=suffix, n_ref=n_ref, n_first_fit=n_first_fit), df))

    # n_curr = 1_000
    # n_ref = 1_000
    # n_first_fit=1_000
    # args_list.append((get_rf_target_evidently_exp(suffix=suffix, n_ref=n_ref, n_first_fit=n_first_fit), df)) # rf + target evidently
    # args_list.append((get_rf_data_evidently_exp(suffix=suffix, n_ref=n_ref, n_first_fit=n_first_fit), df))
    # args_list.append((get_rf_perf_evidently_exp(suffix=suffix, n_ref=n_ref, n_first_fit=n_first_fit), df))
    return args_list

def main():
    args_list = compose_experiments()
    run_concurrent(args_list, NUM_WORKERS)
    
if __name__ == "__main__":
    main()
