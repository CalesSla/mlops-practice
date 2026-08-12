import pandas as pd
import pytest
from preprocessing import preprocess_data


def test_filtering_invalid_rows(sample_data):
    preprocessed_data = preprocess_data(sample_data)
    assert preprocessed_data.shape[0] == 2  # Only 2 valid rows should remain
    assert all(
        preprocessed_data["distance"] >= 0
    )  # All distances should be non-negative
    assert all(
        preprocessed_data["passenger_count"] > 0
    )  # All passenger counts should be positive


def test_original_dataset_unchanged(sample_data):
    original_data_copy = sample_data.copy()
    preprocess_data(sample_data)
    pd.testing.assert_frame_equal(
        sample_data, original_data_copy
    )  # Ensure original data is unchanged


def test_distance_calculation(sample_data):
    preprocessed_data = preprocess_data(sample_data)
    expected_distances = [
        ((1 - 5) ** 2 + (1 - 5) ** 2) ** 0.5,  # For the first valid row
        ((2 - 4) ** 2 + (2 - 4) ** 2) ** 0.5,  # For the second valid row
    ]
    assert all(
        abs(preprocessed_data["distance"].iloc[i] - expected_distances[i]) < 1e-6
        for i in range(len(expected_distances))
    )


def test_preprocessing_returns_correct_columns(sample_data):
    preprocessed_data = preprocess_data(sample_data)
    assert list(preprocessed_data.columns) == [
        "distance",
        "passenger_count",
    ]  # Check for correct columns


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        ({"fare_amount": [10], "passenger_count": [0]}, 0),  # Invalid passenger_count
        (
            {"fare_amount": [10, 20], "passenger_count": [1, 2]},
            2,
        ),  # Valid passenger_count
        ({"fare_amount": [-5], "passenger_count": [1]}, 0),  # Invalid fare_amount
        ({"fare_amount": [10], "passenger_count": [7]}, 0),  # Invalid passenger_count
        ({"fare_amount": [10], "passenger_count": [6]}, 1),  # Valid passenger_count
        ({"fare_amount": [0], "passenger_count": [1]}, 1),  # Edge case: fare_amount = 0
    ],
)
def test_edge_cases_preprocessing(test_input, expected_output):
    base_data = {
        "pickup_longitude": [-73.9],
        "pickup_latitude": [40.7],
        "dropoff_longitude": [-74.0],
        "dropoff_latitude": [40.8],
    }
    num_rows = len(test_input["fare_amount"])
    df_data = {**{k: [v[0]] * num_rows for k, v in base_data.items()}, **test_input}
    df = pd.DataFrame(df_data)
    preprocessed_data = preprocess_data(df)
    assert (
        preprocessed_data.shape[0] == expected_output
    )  # Check if the number of valid rows matches expected output
