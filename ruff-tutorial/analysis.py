import pandas as pd


def clean(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna()
    return data
