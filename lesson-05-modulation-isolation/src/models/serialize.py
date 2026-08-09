import pickle
from pathlib import Path

from sklearn.linear_model import LinearRegression


def serialize_model(model: LinearRegression, file_path: Path) -> None:
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
