import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
from config import Config, ModelParams
from pathlib import Path


def load_config() -> Config:
    with open("config.json") as f:
        return Config.parse_obj(json.load(f))


def load_data(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError as e:
        raise e


def preprocess(data: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.Series]:
    if data is None:
        raise ValueError("Data is None")
    data = data.dropna()
    X = data[config.features]
    y = data[config.target_column]
    return X, y


def train(X: pd.DataFrame, y: pd.Series, params: ModelParams) -> RandomForestRegressor:
    if X is None or y is None:
        raise ValueError("Features or target is None")
    model = RandomForestRegressor(**params.dict())
    model.fit(X, y)
    return model


def save_model(model: RandomForestRegressor, path: Path) -> None:
    if model is not None:
        with open(path, "wb") as f:
            pickle.dump(model, f)


def main() -> None:
    config = load_config()
    data = load_data(config.data_path)
    X, y = preprocess(data, config)

    if X is not None and y is not None:
        X_train, _, y_train, _ = train_test_split(
            X, y, random_state=config.model_params.random_state
        )
        model = train(X_train, y_train, config.model_params)
        save_model(model, config.output_path)
        print("Model saved successfully")


if __name__ == "__main__":
    main()
