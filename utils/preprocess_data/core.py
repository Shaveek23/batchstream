import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder



def ohe_cat_features(df, cat_feature_list):
    for col in cat_feature_list:
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
        transformed = enc.fit_transform(df[[col]])
        transformed_df = pd.DataFrame(
            index=df.index,
            data=transformed, 
            columns=list(enc.get_feature_names_out())
        )
        df.drop(columns=[col], inplace=True)
        df = pd.concat([df, transformed_df], axis=1)
    return df

def encode_ordinal_features(df, ordinal_features_dict):
    for col_name, col_categories in ordinal_features_dict.items():
        enc = OrdinalEncoder(categories=[col_categories])
        transformed = enc.fit_transform(df[[col_name]])
        transformed_df = pd.DataFrame(
            index=df.index,
            data=transformed, 
            columns=list(enc.get_feature_names_out())
        )
        df.drop(columns=[col_name], inplace=True)
        df = pd.concat([df, transformed_df], axis=1)
    return df
