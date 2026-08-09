import pandas as pd


def clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    df = raw_data.dropna()
    return df
