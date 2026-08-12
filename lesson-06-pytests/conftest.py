import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Создает фикстуру с тестовыми данными."""
    data = {
        "fare_amount": [
            10,
            20,
            -5,
            30,
            0,
        ],  # Invalid fare_amount values included for testing
        "passenger_count": [
            1,
            2,
            3,
            0,
            7,
        ],  # Invalid passenger_count values included for testing
        "dropoff_longitude": [1, 2, 3, 4, 5],
        "pickup_longitude": [5, 4, 3, 2, 1],
        "dropoff_latitude": [1, 2, 3, 4, 5],
        "pickup_latitude": [5, 4, 3, 2, 1],
    }
    return pd.DataFrame(data)
