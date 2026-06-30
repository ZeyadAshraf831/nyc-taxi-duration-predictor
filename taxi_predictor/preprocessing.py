import numpy as np
import pandas as pd

from taxi_predictor.config import BINARY_FEATURES, FEATURE_COLUMNS, SCALE_COLS, TRAIN_CSV_PATH
from taxi_predictor.feature_engineering import engineer_features_frame, engineer_inference_frame, haversine

TEST_CSV_PATH = TRAIN_CSV_PATH.parent / "test.csv"
SUBMISSION_PATH = TRAIN_CSV_PATH.parent / "submission.csv"


def clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
    data["dropoff_datetime"] = pd.to_datetime(data["dropoff_datetime"])

    data = data[
        (data["pickup_longitude"].between(-74.25, -73.70))
        & (data["pickup_latitude"].between(40.50, 40.90))
        & (data["dropoff_longitude"].between(-74.25, -73.70))
        & (data["dropoff_latitude"].between(40.50, 40.90))
    ]
    data = data[data["passenger_count"].between(1, 6)]
    data = data[data["trip_duration"].between(60, 7200)]

    data["distance_km"] = haversine(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )
    data = data[data["distance_km"] > 0.1]

    return data.reset_index(drop=True)


def prepare_training_data(
    df: pd.DataFrame,
    scale_cols: list[str] | None = None,
    feature_columns: list[str] | None = None,
):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    feature_columns = feature_columns or FEATURE_COLUMNS
    scale_cols = scale_cols or [col for col in feature_columns if col not in BINARY_FEATURES]
    cleaned = clean_training_frame(df)
    featured = engineer_features_frame(cleaned)

    x = featured[feature_columns].copy()
    y = featured["target"].copy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train[scale_cols] = scaler.fit_transform(x_train[scale_cols])
    x_test[scale_cols] = scaler.transform(x_test[scale_cols])

    return x_train, x_test, y_train, y_test, scaler


def clean_inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])

    data = data[
        (data["pickup_longitude"].between(-74.25, -73.70))
        & (data["pickup_latitude"].between(40.50, 40.90))
        & (data["dropoff_longitude"].between(-74.25, -73.70))
        & (data["dropoff_latitude"].between(40.50, 40.90))
    ]
    data = data[data["passenger_count"].between(1, 6)]

    data["distance_km"] = haversine(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )
    data = data[data["distance_km"] > 0.1]

    return data.reset_index(drop=True)


def prepare_test_features(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_inference_frame(df)
    return engineer_inference_frame(cleaned)
