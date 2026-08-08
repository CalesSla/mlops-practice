import json


def load_params(path: str) -> dict[str, float]:
    with open(path) as f:
        params: dict[str, float] = json.load(f)
        return params


def scale(lr: float, factor: float = 2) -> float:
    return lr * factor
