import json
import os

import pandas as pd

from mess import load_config


def main():
    print(os.getcwd())
    print(json.dumps({"a": 1}))
    print(pd.__version__)
    print(load_config())
