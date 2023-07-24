import os
import pandas as pd
from .core import encode_ordinal_features, ohe_cat_features



path = r'./data/OHIO/datasets'
dir_list = os.listdir(path)
df0 = pd.read_csv(r'./data/OHIO/Ohio_ALL_datasets.csv', sep=',')
variables_to_drop = [record[1] for record in df0.to_records() if record[2] == 'DROP' and record[1] != 'assn']

df_merged = pd.DataFrame()
for dataset in dir_list:
    dataset_path = os.path.join(path, dataset)
    df = pd.read_csv(dataset_path, sep=';')
    df = df.drop(variables_to_drop, axis=1)
    df_merged = pd.concat([df_merged, df])

df_merged.sort_values(by=['assn', 'arr_hr', 'arr_min'])
df_merged = df_merged.drop(['assn'], axis=1)
df_merged.reset_index(drop=True, inplace=True)

df_merged['target'] = df_merged.pop('mode')
df_merged.dropna(subset=['target'], how='any', inplace=True)
df_merged = df_merged[df_merged['target'] != 99.0]
df_merged = df_merged[df_merged['target'] != 97.0]
df_merged['target'] = df_merged['target'] - 11.0
df_merged['target'].astype(dtype=int)

features_to_be_removed = ['r_depart', 'visitors', 'hrshome', 'trp_act4', 'times', 'length', 'morejobs', 'w2_addr', 'occup2', 'indust2', 'v_addr', 'trp_act3', 'studattn', 'school', 's_addr', 'hh_mem', 'nonhh', 'trp_act2']
df_merged.drop(features_to_be_removed, axis=1, inplace=True)

features_na_to_cat = {
    'occup': 99.0, # RF
    'industry': 99.0, # RF
    'primact': 99.0, # RF
    'workhome': 9.0, # DK/RF
    'jobs': 9.0, # RF
    'lic': 9.0, # RF
    'trbrveh': 0.0, 
    'borveh': 0.0
}
df_merged = df_merged.fillna(features_na_to_cat)

for col in ['w1_addr', 'party', 'trpdur']:
    df_merged[col] = df_merged[col].fillna(df_merged[col].mode()[0])

cat_feature_list = ['occup', 'industry', 'workhome', 'jobs', 'primact', 'lic', 'ptype', 'pl_type', 'trp_act1', 'resp', 'relation', 'gender', 'student', 'volunteer', 'trvldywk', 'guest', 'resty', 'own', 'income', 'no_phone']

df_merged = ohe_cat_features(df_merged, cat_feature_list)

df_merged.to_csv(rf'./data/OHIO/ohio_merged.csv', sep=';', index=False)
