import pandas as pd



def get_comobility_dataset():
    df = pd.read_table('./data/comobility/optima.dat')
    df = df.drop(columns=['ID', 'Weight'])[df['Choice'] != -1].reset_index(drop=True)
    df['target'] = df.pop('Choice')
    df['dataset'] = 'comobility'
    return df
