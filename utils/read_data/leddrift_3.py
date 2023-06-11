import pandas as pd
from io import BytesIO, TextIOWrapper
from zipfile import ZipFile
from scipy.io.arff import loadarff
from os import path
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder



def get_leddrift4_df(data_path='./data') -> Tuple[pd.DataFrame, pd.Series]:
    zip_path = path.join(data_path, 'LEDDrift_3/ledDriftSmall.arff.zip')
    with (ZipFile(zip_path, 'r')) as zfile:
        in_mem_fo = TextIOWrapper(BytesIO(zfile.read('ledDriftSmall.arff')), encoding='ascii')
        data = loadarff(in_mem_fo)
        df = pd.DataFrame(data[0])
        to_convert_df = df.select_dtypes([object])
        to_convert_col_names = to_convert_df.columns
        df[to_convert_col_names] = to_convert_df.stack().str.decode('utf-8').unstack()
        df['target'] = df.pop('class')
        df = df.astype(str).astype(int)
        df['dataset'] = 'LEDDrift_3'
    return df
