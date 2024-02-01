import pandas as pd
from .core import ohe_cat_features, encode_ordinal_features


path = r'./data/NTS/nts_data_converted.csv'

df0 = pd.read_csv(r'./data/NTS/NTS.csv', sep=',')
variables_to_drop = [record[1] for record in df0.to_records() if record[2] == 'DROP']

df = pd.read_csv(path, sep=',')
df = df.drop(variables_to_drop, axis=1)
df['target'] = df.pop('mode_main')
df['target'] = df['target'].astype("category").cat.codes

cat_columns = ['male', 'ethnicity', 'license', 'weekend']
df = ohe_cat_features(df, cat_columns)
ordinal_columns = {'education' : ['lower', 'middle', 'higher'], 'income': ['less20', '20to40', 'more40']}
df = encode_ordinal_features(df, ordinal_columns)


df.to_csv(rf'./data/NTS/NTS_preprocessed.csv', sep=';', index=False)
