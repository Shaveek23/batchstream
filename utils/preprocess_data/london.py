import pandas as pd
from .core import ohe_cat_features
from sklearn.preprocessing import FunctionTransformer
import numpy as np



def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
	return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

def transform_cyclic_feature(df, col, period):
	df[f'{col}_sin'] = sin_transformer(period).fit_transform(df[col])
	df[f'{col}_cos'] = cos_transformer(period).fit_transform(df[col])
	df.pop(col)
	return df



path = r'./data/LONDON/london_data_extended_by_mg.csv'

df0 = pd.read_csv(r'./data/LONDON/LPMC_London.csv', sep=',')
variables_to_drop = [record[1] for record in df0.to_records() if record[2] == 'DROP']

df = pd.read_csv(path)
df = df.sort_values(by=['travel_year', 'travel_month', 'travel_date', 'start_time'])
df.reset_index(drop=True, inplace=True)
df = df.drop(variables_to_drop, axis=1)
df['target'] = df.pop('travel_mode')
df['target'] = df['target'] - 1

cat_feature_list = ["purpose", "fueltype", "faretype", "car_ownership"]

df = ohe_cat_features(df, cat_feature_list)

df = transform_cyclic_feature(df, 'travel_date', 31)
df = transform_cyclic_feature(df, 'day_of_week', 7)
df = transform_cyclic_feature(df, 'travel_month', 12)
df = transform_cyclic_feature(df, 'start_time', 24)

df.to_csv(rf'./data/LONDON/london_preprocessed.csv', sep=';', index=False)
