from batchstream.utils.concurrent import run_concurrent
from experiments.first.online.online import get_naive_bayes_online_exp, get_hoeffding_tree_exp, get_srp_exp, get_arf_exp
from utils.read_data.get_dataset import get_dataset
import uuid



NUM_WORKERS = 5

def compose_experiments(dataset_name: str):

    suffix = f"{str(uuid.uuid4())[:4]}_{dataset_name}"
    df = get_dataset(dataset_name)
    window_size = 100

    args_list = []
    args_list.append((get_naive_bayes_online_exp(suffix=suffix, window_size=window_size), df))
    args_list.append((get_hoeffding_tree_exp(suffix=suffix, window_size=window_size), df))
    args_list.append((get_srp_exp(suffix=suffix, window_size=window_size), df))
    args_list.append((get_arf_exp(suffix=suffix, window_size=window_size), df))
    return args_list

def main():
    args_list = compose_experiments(dataset_name='elec')
    run_concurrent(args_list, NUM_WORKERS)
    
if __name__ == "__main__":
    main()
