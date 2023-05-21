import csv
from os import path
import json
import os
from river.datasets import synth



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
    
def download_insects()
