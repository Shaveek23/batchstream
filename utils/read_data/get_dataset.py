from .airlines import get_airlines_df
from .elec import get_elec_df
from .internet_ads import get_internet_ads_df
from .covtype import get_covtype_df



def get_dataset(dataset_name: str, data_path='./data'):

    if 'airlines' in dataset_name.lower():
        return get_airlines_df(data_path)

    if 'elec' in dataset_name.lower():
        return get_elec_df(data_path)

    if 'covtype' in dataset_name.lower():
        return get_covtype_df(data_path)

    if 'internet' in dataset_name.lower():
        return get_internet_ads_df(data_path)
    
    raise ValueError("Dataset not found.")
