import pandas as pd


def validate_data(df: pd.DataFrame, required_columns: list) -> None:
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
