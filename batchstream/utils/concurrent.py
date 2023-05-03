from multiprocessing import Pool
import pandas as pd
from batchstream.experiment.experiment import StreamExperiment


def run_experiment(experiment_pipeline: StreamExperiment, df: pd.DataFrame):
    r''' run_experiment performs a given experiment and logs artifacts.
    Args:
        experiment_pipeline (StreamExperiment): An experiment to be conducted. 
        df (pd.DataFrame): A data frame with data and dataset name
    '''
    try:
        print("START - experiment")
        experiment_pipeline.run(df)
    except Exception as e:
        print("Exception:")
        print(e)
    print("END - experiment")

def run_concurrent(args_list, workers: int=3):
    r''' run_concurrent calls `run_experiment` with defined experiments using `multiprocessing.Pool`
    Args:
        args_list: list of tuples. Each tuple defines arguments passed to the `run_experiment` function, i.e.: a `StreamExperiment` object and a data frame. 
        workers (int): number of processes in a pool to be used
    '''
    results = []
    with Pool(workers) as pool:
        res = pool.starmap_async(run_experiment, args_list)
        results.append(res)

        for res in results:
            res.wait()
