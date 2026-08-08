import pandas as pd  # noqa: I001
import os  # noqa: I001
import sys  # noqa: F401, I001
import json  # noqa: F401, I001
from typing import Dict  # noqa: F401, I001, UP035


def prepare(data: pd.DataFrame) -> Dict:  # noqa: UP006
    temp = data.copy()  # noqa: F841
    data.dropna(inplace=True)  # noqa: PD002
    stats = {
        "rows": len(data),
        "columns": len(data.columns),
        "memory_usage_bytes": int(data.memory_usage(deep=True).sum()),
        "source": os.getcwd(),
    }  # noqa: E501
    return stats
