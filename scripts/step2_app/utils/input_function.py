import pandas as pd

def get_input_features_df(input_features):
    PATH="./ift6758/data/nhl/csv/processed/test_sample.csv"
    df=pd.read_csv(PATH)
    return df[input_features].copy()