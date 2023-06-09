from .airlines import get_airlines_df
from .elec import get_elec_df
from .internet_ads import get_internet_ads_df
from .insects_abrupt import get_insects_abrupt_df
from .covtype import get_covtype_df
from .comobility import get_comobility_dataset
import pandas as pd
from os import path



def get_dataset(dataset_name: str, data_path='./data'):

    if 'airlines' in dataset_name.lower():
        return get_airlines_df(data_path)

    if 'elec' in dataset_name.lower():
        return get_elec_df(data_path)

    if 'covtype' in dataset_name.lower():
        return get_covtype_df(data_path)

    if 'internet' in dataset_name.lower():
        return get_internet_ads_df(data_path)

    if 'insects' in dataset_name.lower() and 'abrupt' in dataset_name.lower():
        return get_insects_abrupt_df(data_path)
    
    if 'rbf' in dataset_name.lower() and '66' in dataset_name.lower():
        df = pd.read_csv(path.join(data_path, 'RBFDrift_0.66_4_4', 'RBFDrift_0.66_4_4.csv'))
        df['dataset'] = 'RBFDrift_0.66'
        return df
    
    if 'stagger' in dataset_name.lower() and '1k' in dataset_name.lower():
        df = pd.read_csv(path.join(data_path, 'STAGGER', 'stagger_1K.csv'))
        df['dataset'] = 'stagger'
        return df
    
    if 'stagger' in dataset_name.lower():
        df = pd.read_csv(path.join(data_path, 'STAGGER', 'stagger_25K.csv'))
        df['dataset'] = 'stagger'
        return df

    if 'led' in dataset_name.lower():
        df = pd.read_csv(path.join(data_path, 'LEDDrift_4_4', 'LEDDrift_4_4.csv'))
        df['dataset'] = 'LEDDrift_4_4'
        return df
    
    if 'comobility' in dataset_name.lower():
        return get_comobility_dataset()

    raise ValueError("Dataset not found.")
