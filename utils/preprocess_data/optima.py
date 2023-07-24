path = r'./data/OPTIMA/optima.csv'

df0 = pd.read_csv(r'./data/OPTIMA/optima_control_file.csv', sep=',')
variables_to_drop = [record[1] for record in df0.to_records() if record[2] == 'DROP']

df = pd.read_csv(path, sep='\t')
df = df.drop(variables_to_drop, axis=1)
df['target'] = df.pop('Choice')
df['target'] += 1

df.to_csv(rf'./data/OPTIMA/optima_preprocessed.csv', sep=';', index=False)
