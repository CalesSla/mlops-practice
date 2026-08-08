from typing import Final, Literal


def load_data(path: str, fmt: Literal["csv", "parquet"]) -> None:
    return None


csv_type: Final = "csv"

load_data("data.csv", "csv")  # valid
# load_data("data.xlsx", "xlsx") # invalid, mypy will catch this
load_data("data.csv", csv_type)
