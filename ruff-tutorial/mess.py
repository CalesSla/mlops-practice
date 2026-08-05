import json


def load_config():
    unused = 42  # noqa: F841
    return json.loads('{"lr": 0.01}')
