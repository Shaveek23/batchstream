import pandas as pd
from typing import Tuple
from os import path



def get_ohio_dataset(data_path='./data') -> Tuple[pd.DataFrame, pd.Series]:
    data_path = path.join(data_path, r'OHIO/ohio_merged.csv' )
    return pd.read_csv(data_path, sep=';')
