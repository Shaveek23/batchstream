import pandas as pd
from io import BytesIO, TextIOWrapper
from zipfile import ZipFile
from scipy.io.arff import loadarff
from os import path
from typing import Tuple
from sklearn import preprocessing



def get_insects_abrupt_df(data_path='./data') -> Tuple[pd.DataFrame, pd.Series]:
    zip_path = path.join(data_path, 'Insects/INSECTS-abrupt_balanced_norm.zip')
    with (ZipFile(zip_path, 'r')) as zfile:
        in_mem_fo = TextIOWrapper(BytesIO(zfile.read('INSECTS-abrupt_balanced_norm.arff')), encoding='ascii')
        data = loadarff(in_mem_fo)
        df = pd.DataFrame(data[0])
        to_convert_df = df.select_dtypes([object])
        to_convert_col_names = to_convert_df.columns
        df[to_convert_col_names] = to_convert_df.stack().str.decode('utf-8').unstack()
        le = preprocessing.LabelEncoder()
        target = le.fit_transform(df['class'])
        df.pop('class')
        df['target'] = target
        df['dataset'] = 'insects_abrupt'
    return df
    