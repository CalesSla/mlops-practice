# src/preprocessing.py
import numpy as np
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Выполняет предобработку данных о поездках Uber."""
    # 1. Фильтрация по стоимости и пассажирам
    df_internal = df.copy()
    mask = (
        (df_internal["fare_amount"] >= 0)
        & (df_internal["passenger_count"] > 0)
        & (df_internal["passenger_count"] <= 6)
    )

    df_internal.drop(index=df_internal.loc[~mask].index, inplace=True)

    # 2. Создание нового признака 'distance'
    df_internal["distance"] = np.sqrt(
        (df_internal["dropoff_longitude"] - df_internal["pickup_longitude"]) ** 2
        + (df_internal["dropoff_latitude"] - df_internal["pickup_latitude"]) ** 2
    )
    final_features = df_internal[["distance", "passenger_count"]]
    return final_features
