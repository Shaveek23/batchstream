import pandas as pd
from io import BytesIO, TextIOWrapper
from zipfile import ZipFile
from scipy.io.arff import loadarff
from os import path
from typing import Tuple



def get_internet_ads_df(data_path='./data') -> Tuple[pd.DataFrame, pd.Series]:
    zip_path = path.join(data_path, 'ADS/internet_ads.arff.zip')
    with (ZipFile(zip_path, 'r')) as zfile:
        in_mem_fo = TextIOWrapper(BytesIO(zfile.read('internet_ads.arff')), encoding='ascii')
        data = loadarff(in_mem_fo)
        df = pd.DataFrame(data[0])
        to_convert_df = df.select_dtypes([object])
        to_convert_col_names = to_convert_df.columns
        df[to_convert_col_names] = to_convert_df.stack().str.decode('utf-8').unstack()
        class_col = df.pop('class').replace(['noad', 'ad'], [0, 1])
        df[df.columns[3:]] = df.loc[:, "local":].astype(str).astype(int)
        df['target'] = class_col.copy()
        df['dataset'] = 'internet_ads'
    return df