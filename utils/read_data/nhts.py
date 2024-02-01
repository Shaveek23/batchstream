import pandas as pd
from typing import Tuple
from os import path



def get_nhts_dataset(state_group_type, data_path='./data'):
    data_path = path.join(data_path, rf'NHTS/NHTS_{state_group_type}_merged.parquet.gzip')
    df = pd.read_parquet(data_path) 
    df = df.reset_index(drop=True)
    df['dataset'] = f'NHTS_{state_group_type}'
    return df
