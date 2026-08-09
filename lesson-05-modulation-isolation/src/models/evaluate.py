import pandas as pd
from sklearn.linear_model import LinearRegression


def evaluate_model(
    model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    score = model.score(X_test, y_test)
    print(f"Model R^2 score: {score}")
