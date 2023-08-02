import pandas as pd
from utils.preprocess_data.core import ohe_cat_features

path = r'./data/OPTIMA/optima.csv'

df0 = pd.read_csv(r'./data/OPTIMA/optima_control_file.csv', sep=',')
variables_to_drop = [record[1] for record in df0.to_records() if record[2] == 'DROP']

df = pd.read_csv(path, sep='\t')
df = df.drop(variables_to_drop, axis=1)
df['target'] = df.pop('Choice')
df['target'] += 1


cat_columns = ['DestAct', 'LangCode', 'ModeToSchool', 'ResidChild', 'Internet', 'NewsPaperSubs', 'HouseType', 'OwnHouse', 'Gender', 'Mothertongue', 'FamilSitu', 'OccupStat', 'SocioProfCat', 'HalfFareST', 'LineRelST', 'GenAbST', 'AreaRelST', 'OtherST', 'Education', 'Envir01',
 'Envir02',
 'Envir03',
 'Envir04',
 'Envir05',
 'Envir06',
 'Mobil01',
 'Mobil02',
 'Mobil03',
 'Mobil04',
 'Mobil05',
 'Mobil06',
 'Mobil07',
 'Mobil08',
 'Mobil09',
 'Mobil10',
 'Mobil11',
 'Mobil12',
 'Mobil13',
 'Mobil14',
 'Mobil15',
 'Mobil16',
 'Mobil17',
 'Mobil18',
 'Mobil19',
 'Mobil20',
 'Mobil21',
 'Mobil22',
 'Mobil23',
 'Mobil24',
 'Mobil25',
 'Mobil26',
 'Mobil27',
 'ResidCh01',
 'ResidCh02',
 'ResidCh03',
 'ResidCh04',
 'ResidCh05',
 'ResidCh06',
 'ResidCh07',
 'LifSty01',
 'LifSty02',
 'LifSty03',
 'LifSty04',
 'LifSty05',
 'LifSty06',
 'LifSty07',
 'LifSty08',
 'LifSty09',
 'LifSty10',
 'LifSty11',
 'LifSty12',
 'LifSty13',
 'LifSty14'
 ]

df = ohe_cat_features(df, cat_columns)
df.to_csv(rf'./data/OPTIMA/optima_preprocessed.csv', sep=';', index=False)
