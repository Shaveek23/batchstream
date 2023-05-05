from batchstream.utils.concurrent import run_concurrent
from experiments.covtype.online.online import get_naive_bayes_online_exp, get_hoeffding_tree_exp, get_srp_exp, get_arf_exp
from utils.read_data.covtype import get_covtype_dataset
import uuid



NUM_WORKERS = 5

def compose_experiments():


    suffix = str(uuid.uuid4())[:4]
    df = get_covtype_dataset()
    args_list = []
    args_list.append((get_naive_bayes_online_exp(suffix=suffix), df))
    args_list.append((get_hoeffding_tree_exp(suffix=suffix), df))
    args_list.append((get_srp_exp(suffix=suffix), df))
    args_list.append((get_arf_exp(suffix=suffix), df))
    return args_list

def main():
    args_list = compose_experiments()
    run_concurrent(args_list, NUM_WORKERS)
    
if __name__ == "__main__":
    main()
