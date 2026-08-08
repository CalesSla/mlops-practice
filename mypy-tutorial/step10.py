from typing import Protocol


class Preprocessor(Protocol):
    def transform(self, values: list[float]) -> list[float] | list[str]: ...


class Scaler:
    def transform(self, values: list[float]) -> list[float]:
        return [v * 2 for v in values]


class Tokenizer:
    def transform(self, values: list[float]) -> list[float]:
        return [v for v in values]


def run(p: Preprocessor, values: list[float]) -> list[float] | list[str]:
    return p.transform(values)


run(Scaler(), [1.0, 2.0, 3.0])
run(Tokenizer(), [1.0, 2.0, 3.0])
