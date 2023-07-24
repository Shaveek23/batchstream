path = r'./data/LONDON/london_data_extended_by_mg.csv'

df0 = pd.read_csv(r'./data/LONDON/LPMC_London.csv', sep=',')
variables_to_drop = [record[1] for record in df0.to_records() if record[2] == 'DROP']

df = pd.read_csv(path)
df.sort_values(by=['travel_year', 'travel_month', 'travel_date', 'start_time'])
df.reset_index(drop=True, inplace=True)
df = df.drop(variables_to_drop, axis=1)
df['target'] = df.pop('travel_mode')
df['target'] = df['target'] - 1

df.to_csv(rf'./data/LONDON/london_preprocessed.csv', sep=';', index=False)
