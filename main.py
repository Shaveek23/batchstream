from batchstream.utils.concurrent import run_concurrent
from experiments.covtype.covtype_rf_adwin_only import get_covtype_rf_adwin_only_exp
from experiments.covtype.covtype_rf_data_evidently import get_rf_data_evidently_exp
from experiments.covtype.covtype_rf_target_evidently import get_rf_target_evidently_exp
from experiments.covtype.covtype_rf_perf_evidently import get_rf_perf_evidently_exp
from utils.read_data.covtype import get_covtype_dataset



NUM_WORKERS = 10

def compose_experiments():
    df = get_covtype_dataset()
    args_list = []
    args_list.append((get_covtype_rf_adwin_only_exp(df), df)) # rf + adwin (data + target)
    args_list.append((get_rf_target_evidently_exp(), df)) # rf + target evidently
    args_list.append((get_rf_data_evidently_exp(), df))
    args_list.append((get_rf_perf_evidently_exp(), df))
    return args_list

def main():
    args_list = compose_experiments()
    run_concurrent(args_list, NUM_WORKERS)
    
if __name__ == "__main__":
    main()
