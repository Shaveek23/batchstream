import pandas as pd
from typing import Tuple
from os import path



def get_nhts_dataset(data_path='./data') -> Tuple[pd.DataFrame, pd.Series]:
    data_path = path.join(data_path, r'NHTS/NHTS_preprocessed.csv' )
    df = pd.read_csv(data_path, sep=';')
    df['dataset'] = 'NHTS'
    return df
