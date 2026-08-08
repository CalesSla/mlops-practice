from collections.abc import Callable


def preprocess(values: list[float], transform: Callable[[float], float]) -> list[float]:
    return [transform(v) for v in values]


def scale(x: float) -> float:
    return x * 2


def label(x: float) -> str:
    return f"Value: {x!s}"


preprocessed_values = preprocess([1.0, 2.0, 3.0], scale)
# preprocessed_labels = preprocess([1.0, 2.0, 3.0], label)
preprocess_with_lambda = preprocess([1.0, 2.0, 3.0], lambda x: x * 2)
