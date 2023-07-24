import pandas as pd



path = r'./data/NTS/nts_data_converted.csv'

df0 = pd.read_csv(r'./data/NTS/NTS.csv', sep=',')
variables_to_drop = [record[1] for record in df0.to_records() if record[2] == 'DROP']

df = pd.read_csv(path, sep=',')
df = df.drop(variables_to_drop, axis=1)
df['target'] = df.pop('mode_main')
for col in ['male', 'ethnicity', 'education', 'income', 'license', 'weekend', 'target']:
    df[col] = df[col].astype("category").cat.codes

df.to_csv(rf'./data/NTS/NTS_preprocessed.csv', sep=';', index=False)
