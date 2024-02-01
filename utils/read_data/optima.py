import pandas as pd
from typing import Tuple
from os import path



def get_optima_dataset(data_path='./data') -> Tuple[pd.DataFrame, pd.Series]:
    data_path = path.join(data_path, r'OPTIMA/optima_preprocessed.parquet.gzip')
    df = pd.read_parquet(data_path) 
    df['dataset'] = 'OPTIMA'
    return df
