import pandas as pd
from sklearn.linear_model import LinearRegression


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
