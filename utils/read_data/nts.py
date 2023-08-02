import pandas as pd
from typing import Tuple
from os import path



def get_nts_dataset(data_path='./data') -> Tuple[pd.DataFrame, pd.Series]:
    data_path = path.join(data_path, r'NTS/NTS_preprocessed.csv')
    df = pd.read_csv(data_path, sep=';')
    df['dataset'] = 'NTS'
    return df
