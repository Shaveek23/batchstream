import pandas as pd
from io import BytesIO, TextIOWrapper
from zipfile import ZipFile
from scipy.io.arff import loadarff
from os import path
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder



def get_airlines_df(data_path='./data') -> Tuple[pd.DataFrame, pd.Series]:
    zip_path = path.join(data_path, 'AIRLINES/airlines.arff.zip')
    with (ZipFile(zip_path, 'r')) as zfile:
        in_mem_fo = TextIOWrapper(BytesIO(zfile.read('airlines.arff')), encoding='ascii')
        data = loadarff(in_mem_fo)
        df = pd.DataFrame(data[0])
        to_convert_df = df.select_dtypes([object])
        to_convert_col_names = to_convert_df.columns
        df[to_convert_col_names] = to_convert_df.stack().str.decode('utf-8').unstack()
        class_col = df.pop('Delay')
        for col in ['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek']:
            enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            transformed = enc.fit_transform(df[[col]])
            transformed_df = pd.DataFrame(
                index=df.index,
                data=transformed, 
                columns=list(enc.get_feature_names_out())
            )
            df.drop(columns=[col], inplace=True)
            df = pd.concat([df, transformed_df], axis=1)
        df['target'] = class_col.copy()
        df = df.sample(frac=1, random_state=125).reset_index(drop=True)
        df['dataset'] = 'airlines'
    return df
