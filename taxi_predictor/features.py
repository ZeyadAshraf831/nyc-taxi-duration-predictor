from datetime import datetime

import numpy as np
import pandas as pd
import torch

from taxi_predictor.config import BINARY_FEATURES, FEATURE_COLUMNS, LEGACY_FEATURE_COLUMNS, LEGACY_SCALE_COLS
from taxi_predictor.feature_engineering import build_feature_row, haversine


def build_features(
    pickup_lat: float,
    pickup_lon: float,
    dropoff_lat: float,
    dropoff_lon: float,
    vendor_id: int,
    passenger_count: int,
    pickup_datetime: datetime,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    columns = columns or FEATURE_COLUMNS
    feature_dict = build_feature_row(
        pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, vendor_id, passenger_count, pickup_datetime
    )

    features = pd.DataFrame([{name: feature_dict[name] for name in columns}])
    meta = {
        "distance_km": feature_dict["distance_km"],
        "is_rush_hour": feature_dict["is_rush_hour"],
        "is_weekend": feature_dict["is_weekend"],
    }
    return features, meta


def scale_columns_for_features(columns: list[str]) -> list[str]:
    if columns == LEGACY_FEATURE_COLUMNS:
        return LEGACY_SCALE_COLS
    return [col for col in columns if col not in BINARY_FEATURES]


def predict_duration(model, scaler, features: pd.DataFrame, columns: list[str] | None = None) -> float:
    columns = columns or list(features.columns)
    scale_cols = scale_columns_for_features(columns)

    scaled = features.copy()
    scaled[scale_cols] = scaler.transform(scaled[scale_cols])

    tensor = torch.tensor(scaled[columns].values, dtype=torch.float32)
    with torch.no_grad():
        return float(np.expm1(model(tensor).item()))
