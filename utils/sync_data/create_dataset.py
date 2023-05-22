import csv
from os import path
import json
import os
from river.datasets import synth
import pandas as pd
from sklearn.preprocessing import OneHotEncoder



def save_dataset(dataset, dataset_name, n_samples, data_dir):
    dir_path = path.join(data_dir, dataset_name)
    if not path.exists(dir_path):
        os.mkdir(dir_path)
    df_path = path.join(dir_path, f'{dataset_name}.csv')
    info_path = path.join(dir_path, 'info.json')
    with open(df_path, 'w', newline='', encoding='utf-8') as f:
        first = True
        for x, y in dataset.take(n_samples):
            x.update({'target': y})
            w = csv.DictWriter(f, x.keys())
            if first: w.writeheader()
            w.writerow(x)
            first = False
    with open(info_path, 'w') as f:
        json.dump(dataset._get_params(), f)

def generate_RBFdrift(n_samples=500_000, n_classes=4, n_features=4, n_centroids=20,
    change_speed=0.66, n_drift_centroids=10, seed_model=42, seed_sample=42, data_dir='./data'):

    dataset = synth.RandomRBFDrift(seed_model=42, seed_sample=42,
        n_classes=4, n_features=4, n_centroids=20,
        change_speed=0.66, n_drift_centroids=10
    )
    dataset_name = f'RBFDrift_{change_speed}_{n_features}_{n_classes}'
    save_dataset(dataset, dataset_name, n_samples, data_dir)

def generate_LEDdrift(n_samples=200_000, seed=42, noise_percentage=0.20,
        irrelevant_features=False, n_drift_features=4, data_dir='./data'):
    dataset = synth.LEDDrift(seed, noise_percentage, irrelevant_features, n_drift_features)
    dataset_name = f'LEDDrift_{n_drift_features}_{n_drift_features}'
    save_dataset(dataset, dataset_name, n_samples, data_dir)
    
def generate_stagger_dataset(seed=42, balance_classes=True, drift_step: int=25_000, data_dir='./data'):
    X = []
    Y = []
    dataset = synth.STAGGER(classification_function=0, seed=seed, balance_classes=balance_classes)
    for x, y in dataset.take(drift_step):
        X.append(x)
        Y.append(y)

    dataset.classification_function = 1
    for x, y in dataset.take(drift_step):
        X.append(x)
        Y.append(y)

    dataset.classification_function = 2
    for x, y in dataset.take(drift_step):
        X.append(x)
        Y.append(y)

    df = pd.DataFrame(X)

    ohe = OneHotEncoder()

    df = pd.DataFrame(ohe.fit_transform(df).toarray())
    df.columns = ['size_0', 'size_1', 'size_2', 'color_0', 'color_1', 'color_2', 'shape_0', 'shape_1', 'shape_2']

    df['target'] = Y
    
    df.to_csv(f'{data_dir}/STAGGER/stagger_{drift_step // 1000}K.csv', index=False)
    return df
