import pandas as pd



path = r'./data/NHTS/NHTS_data_for_ML.csv'

df0 = pd.read_csv(r'./data/NHTS/NHTS.csv', sep=',')
variables_to_drop = [record[1] for record in df0.to_records() if record[2] == 'DROP']


df = pd.read_csv(path, sep=',')
df = df.sort_values(by=['TDAYDATE', 'STRTTIME'])
df['WRKTIME'] = df['WRKTIME'].str[0:5].str.replace(":", "").astype(int) + df['WRKTIME'].str[-2:].str.replace("PM", "1200").str.replace("AM", "0").astype(int)
df = df.drop(variables_to_drop, axis=1)
df = df.drop(["min_HHSTATE", "max_HHSTATE", "min_HH_CBSA", "max_HH_CBSA", "min_HBHUR", "max_HBHUR", 'max_MAKE',	'min_MAKE', 'min_MODEL', 'max_MODEL'], axis=1)
df['target'] = df.pop('TRPTRANS')
df = df[df['target'] > 0] # rows with negative values should be dropped
df['target'] = df['target'].replace(97, 21)
df['target'] = df['target'] - 1

state_group_type = 'NE'

WEST = ['WA', 'OR', 'CA', 'NV', 'UT', 'AZ', 'NM', 'CO', 'WY', 'MT', 'ID', 'HI', "AK"]
SOUTHWEST = ['TX', 'OK', 'LA', 'AR']
SOUTHEAST = ['MS', 'AL', 'TN', 'KY', 'GA', 'FL', 'SC', 'NC', 'VA', 'WV', 'DE', 'MD', 'DC']
MIDWEST = ['ND', 'SD', 'NE', 'KS', 'MN', 'IA', 'MO', 'WI', 'IL', 'MI', 'IN', 'OH']
NORTHEAST = ['PA', 'NY', 'NJ', 'CT', 'RI', 'MA', 'VT', 'NH', 'ME']

if state_group_type == 'W':
    state_group = WEST
elif state_group_type == 'SE':
    state_group = SOUTHEAST
elif state_group_type == 'SW':
    state_group = SOUTHWEST 
elif state_group_type == 'MW':
    state_group = MIDWEST
elif state_group_type == 'NE':
    state_group = NORTHEAST
else:
    raise ValueError("Unknown state group.")

df = df[df["HHSTATE"].isin(state_group)]

na_columns = list({k: v for k, v in df.isna().sum().to_dict().items() if v != 0 }.keys())
for col in na_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

c_columns = ["ALT_16", "ALT_23", "ALT_45", "BIKE", "BIKE2SAVE", "BIKE_DFR", "BIKE_GKP", "BORNINUS", "BUS", "CAR", "CDIVMSAR", "CENSUS_D", "CENSUS_R", "CONDNIGH", "CONDPUB", "CONDRIDE", "CONDRIVE", "CONDSPEC", "CONDTAX", "CONDTRAV", "DBHTNRNT", "DBHUR", "DBPPOPDN", "DBRESDN", "DIARY", "DRIVER", "DROP_PRK", "DRVR_FLG", "DTEEMPDN", "DTHTNRNT", "DTPPOPDN", "DTRESDN", "EDUC", "FLEXTIME", "FRSTHM17", "GT1JBLWK", "HBHTNRNT", "HBHUR", "HBPPOPDN", "HBRESDN", "HEALTH", "HHFAMINC", "HHMEMDRV", "HHRELATD", "HHRESP", "HHSTATE", "HHSTFIPS", "HH_CBSA", "HH_HISP", "HH_RACE", "HOMEOWN", "HOUSEID", "HTEEMPDN", "HTHTNRNT", "HTPPOPDN", "HTRESDN", "LIF_CYC", "LOOP_TRIP", "LSTTRDAY17", "MEDCOND", "MEDCOND6",  "MSACAT", "MSASIZE", "OBHTNRNT", "OBHUR", "OBPPOPDN", "OBRESDN", "OCCAT", "ONTD_P1", "ONTD_P10", "ONTD_P11", "ONTD_P12", "ONTD_P13", "ONTD_P2", "ONTD_P3", "ONTD_P4", "ONTD_P5", "ONTD_P6", "ONTD_P7", "ONTD_P8", "ONTD_P9", "OTEEMPDN", "OTHTNRNT", "OTPPOPDN", "OTRESDN", "OUTCNTRY", "OUTOFTWN", "PARA", "PAYPROF", "PC", "PERSONID", "PHYACT", "PLACE", "PRICE", "PRMACT", "PROXY", "PSGR_FLG", "PTRANS", "PUBTRANS", "RAIL", "R_HISP", "R_RELAT", "R_SEX", "R_SEX_IMP", "SAMEPLC", "SAMPSTRAT", "SCHTRN1", "SCHTRN2", "SCHTYP", "SCRESP", "SMPLSRCE", "SPHONE", "STRTTIME", "TAB", "TAXI", "TDAYDATE", "TDCASEID", "TDTRPNUM", "TDWKND", "TRACC_BUS", "TRACC_CRL", "TRACC_OTH", "TRACC_POV", "TRACC_SUB", "TRACC_WLK", "TRAIN", "TRAVDAY", "TREGR_BUS", "TREGR_CRL", "TREGR_OTH", "TREGR_POV", "TREGR_SUB", "TREGR_WLK", "TRIPPURP", "TRPHHVEH", "URBAN", "URBANSIZE", "URBRUR", "USEPUBTR", "VEHID", "VEHTYPE", "WALK", "WALK2SAVE", "WALK_DEF", "WALK_GKQ", "WEBUSE17", "WHODROVE", "WHOPROXY", "WHYFROM", "WHYTO", "WHYTRP1S", "WHYTRP90", "WKFTPT", "WKRMHM", "WKSTFIPS", "WORKER", "WRKTIME", "WRKTRANS", "WRK_HOME", "W_CANE", "W_CHAIR", "W_CRUTCH", "W_DOG", "W_MTRCHR", "W_NONE", "W_SCOOTR", "W_WHCANE", "W_WLKR"]

ordinal_cols = ["HBPPOPDN", "HBHTNRNT", "DTEEMPDN", "DTHTNRNT", "DTPPOPDN", "DTRESDN", "HBRESDN", "HTEEMPDN", "HTHTNRNT", "HTPPOPDN", "HTRESDN", "OBHTNRNT", "OBPPOPDN", "OBRESDN", "OTEEMPDN", "OTHTNRNT", "OTPPOPDN", "OTRESDN"]
cat_columns = []
for col in c_columns:
    if col not in ordinal_cols and col not in variables_to_drop:
        cat_columns.append(col)
df = ohe_cat_features(df, cat_columns)
df.to_parquet(rf'./data/NHTS/NTHS_{state_group_type}_merged.parquet.gzip', compression='gzip')  
