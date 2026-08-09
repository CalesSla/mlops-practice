import pandas as pd


def add_wind_humidity_ratio(
    df: pd.DataFrame, wind_col: str, humidity_col: str, wind_humidity_ratio_col: str
) -> pd.DataFrame:
    df_new = df.copy()
    df_new[wind_humidity_ratio_col] = df_new[wind_col] / df_new[humidity_col].replace(
        0, 0.1
    )
    return df_new
