import pandas as pd
from sklearn.model_selection import train_test_split


def split_into_train_and_test(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list,
    random_state: int,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
