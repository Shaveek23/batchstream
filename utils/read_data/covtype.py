import pandas as pd
from io import BytesIO, TextIOWrapper
from zipfile import ZipFile
from scipy.io.arff import loadarff
from os import path
from typing import Tuple



def get_covtype_dataset(data_path='./data') -> Tuple[pd.DataFrame, pd.Series]:
    zip_path = path.join(data_path, 'COVTYPE/covtypeNorm.arff.zip')
    with (ZipFile(zip_path, 'r')) as zfile:
        in_mem_fo = TextIOWrapper(BytesIO(zfile.read('covtypeNorm.arff')), encoding='ascii')
        data = loadarff(in_mem_fo)
        df = pd.DataFrame(data[0])
        to_convert_df = df.select_dtypes([object])
        to_convert_col_names = to_convert_df.columns
        df[to_convert_col_names] = to_convert_df.stack().str.decode('utf-8').unstack()
        class_col = df.pop('class').replace(['noad', 'ad'], [0, 1])
        df['target'] = class_col.copy()
        df.loc[:, "Wilderness_Area1":] = df.loc[:, "Wilderness_Area1":].astype(str).astype(int)
        df['dataset'] = 'covtypeNorm'
    return df
    