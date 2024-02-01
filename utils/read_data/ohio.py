import pandas as pd
from typing import Tuple
from os import path



def get_ohio_dataset(data_path='./data') -> Tuple[pd.DataFrame, pd.Series]:
    data_path = path.join(data_path, r'OHIO/ohio_merged.parquet.gzip' )
    df = pd.read_parquet(data_path) 
    df['dataset'] = 'OHIO_merged'
    return df
