import json
from pathlib import Path


def load_config(config_path: Path = Path("config.json")):
    with open(config_path) as f:
        config = json.load(f)
    return config
