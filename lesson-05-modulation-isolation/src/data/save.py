from pathlib import Path

import pandas as pd


def save_data(df: pd.DataFrame, path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
