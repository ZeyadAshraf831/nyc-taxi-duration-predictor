from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from taxi_predictor.config import (
    AIRPORT_RADIUS_KM,
    JFK_COORDS,
    LGA_COORDS,
    LONG_TRIP_KM,
    NYC_MAP_CENTER,
    RUSH_HOURS,
    SHORT_TRIP_KM,
)

KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON_NYC = 111.0 * np.cos(np.radians(NYC_MAP_CENTER[0]))
MORNING_RUSH_HOURS = [7, 8, 9]
EVENING_RUSH_HOURS = [17, 18, 19]
NIGHT_HOURS = [0, 1, 2, 3, 4, 5, 22, 23]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def encode_vendor(vendor_id: int) -> int:
    return 0 if vendor_id == 1 else 1


def _bearing_sin_cos(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    pickup_lat_rad = np.radians(pickup_lat)
    pickup_lon_rad = np.radians(pickup_lon)
    dropoff_lat_rad = np.radians(dropoff_lat)
    dropoff_lon_rad = np.radians(dropoff_lon)
    delta_lon = dropoff_lon_rad - pickup_lon_rad

    bearing = np.arctan2(
        np.sin(delta_lon) * np.cos(dropoff_lat_rad),
        np.cos(pickup_lat_rad) * np.sin(dropoff_lat_rad)
        - np.sin(pickup_lat_rad) * np.cos(dropoff_lat_rad) * np.cos(delta_lon),
    )
    return np.sin(bearing), np.cos(bearing)


def _near_airport(lat, lon, airport_lat, airport_lon):
    return haversine(lat, lon, airport_lat, airport_lon) <= AIRPORT_RADIUS_KM


def _airport_and_trip_flags(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, distance_km):
    pickup_near_jfk = int(_near_airport(pickup_lat, pickup_lon, JFK_COORDS[0], JFK_COORDS[1]))
    dropoff_near_jfk = int(_near_airport(dropoff_lat, dropoff_lon, JFK_COORDS[0], JFK_COORDS[1]))
    pickup_near_lga = int(_near_airport(pickup_lat, pickup_lon, LGA_COORDS[0], LGA_COORDS[1]))
    dropoff_near_lga = int(_near_airport(dropoff_lat, dropoff_lon, LGA_COORDS[0], LGA_COORDS[1]))
    is_short_trip = int(distance_km < SHORT_TRIP_KM)
    is_long_trip = int(distance_km > LONG_TRIP_KM)
    return {
        "pickup_near_jfk": pickup_near_jfk,
        "dropoff_near_jfk": dropoff_near_jfk,
        "pickup_near_lga": pickup_near_lga,
        "dropoff_near_lga": dropoff_near_lga,
        "is_short_trip": is_short_trip,
        "is_long_trip": is_long_trip,
    }


def build_feature_row(
    pickup_lat: float,
    pickup_lon: float,
    dropoff_lat: float,
    dropoff_lon: float,
    vendor_id: int,
    passenger_count: int,
    pickup_datetime: datetime,
) -> dict:
    dt = pd.to_datetime(pickup_datetime)
    distance_km = float(haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon))

    pickup_hour = int(dt.hour)
    pickup_day = int(dt.day)
    pickup_month = int(dt.month)
    pickup_weekday = int(dt.dayofweek)

    delta_latitude = dropoff_lat - pickup_lat
    delta_longitude = dropoff_lon - pickup_lon
    abs_delta_latitude = abs(delta_latitude)
    abs_delta_longitude = abs(delta_longitude)
    manhattan_km = abs_delta_latitude * KM_PER_DEG_LAT + abs_delta_longitude * KM_PER_DEG_LON_NYC

    bearing_sin, bearing_cos = _bearing_sin_cos(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    pickup_dist_center = float(haversine(pickup_lat, pickup_lon, NYC_MAP_CENTER[0], NYC_MAP_CENTER[1]))
    dropoff_dist_center = float(haversine(dropoff_lat, dropoff_lon, NYC_MAP_CENTER[0], NYC_MAP_CENTER[1]))

    is_rush_hour = int(pickup_hour in RUSH_HOURS)
    is_weekend = int(pickup_weekday >= 5)
    is_night = int(pickup_hour in NIGHT_HOURS)
    is_morning_rush = int(pickup_hour in MORNING_RUSH_HOURS)
    is_evening_rush = int(pickup_hour in EVENING_RUSH_HOURS)
    airport_flags = _airport_and_trip_flags(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, distance_km)

    return {
        "vendor_id": encode_vendor(vendor_id),
        "passenger_count": passenger_count,
        "pickup_longitude": pickup_lon,
        "pickup_latitude": pickup_lat,
        "dropoff_longitude": dropoff_lon,
        "dropoff_latitude": dropoff_lat,
        "distance_km": distance_km,
        "log_distance_km": float(np.log1p(distance_km)),
        "sqrt_distance_km": float(np.sqrt(distance_km)),
        "manhattan_km": float(manhattan_km),
        "delta_latitude": delta_latitude,
        "delta_longitude": delta_longitude,
        "abs_delta_latitude": abs_delta_latitude,
        "abs_delta_longitude": abs_delta_longitude,
        "bearing_sin": float(bearing_sin),
        "bearing_cos": float(bearing_cos),
        "pickup_dist_center": pickup_dist_center,
        "dropoff_dist_center": dropoff_dist_center,
        **airport_flags,
        "hour_sin": float(np.sin(2 * np.pi * pickup_hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * pickup_hour / 24)),
        "weekday_sin": float(np.sin(2 * np.pi * pickup_weekday / 7)),
        "weekday_cos": float(np.cos(2 * np.pi * pickup_weekday / 7)),
        "month_sin": float(np.sin(2 * np.pi * pickup_month / 12)),
        "month_cos": float(np.cos(2 * np.pi * pickup_month / 12)),
        "pickup_day": pickup_day,
        "distance_x_rush": distance_km * is_rush_hour,
        "distance_x_weekend": distance_km * is_weekend,
        "is_rush_hour": is_rush_hour,
        "is_weekend": is_weekend,
        "is_night": is_night,
        "is_morning_rush": is_morning_rush,
        "is_evening_rush": is_evening_rush,
    }


def _add_engineered_columns(data: pd.DataFrame) -> pd.DataFrame:
    pickup_dt = pd.to_datetime(data["pickup_datetime"])

    data["pickup_hour"] = pickup_dt.dt.hour
    data["pickup_day"] = pickup_dt.dt.day
    data["pickup_month"] = pickup_dt.dt.month
    data["pickup_weekday"] = pickup_dt.dt.dayofweek

    data["hour_sin"] = np.sin(2 * np.pi * data["pickup_hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["pickup_hour"] / 24)
    data["weekday_sin"] = np.sin(2 * np.pi * data["pickup_weekday"] / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * data["pickup_weekday"] / 7)
    data["month_sin"] = np.sin(2 * np.pi * data["pickup_month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["pickup_month"] / 12)

    data["is_rush_hour"] = data["pickup_hour"].isin(RUSH_HOURS).astype(int)
    data["is_weekend"] = (data["pickup_weekday"] >= 5).astype(int)
    data["is_night"] = data["pickup_hour"].isin(NIGHT_HOURS).astype(int)
    data["is_morning_rush"] = data["pickup_hour"].isin(MORNING_RUSH_HOURS).astype(int)
    data["is_evening_rush"] = data["pickup_hour"].isin(EVENING_RUSH_HOURS).astype(int)

    data["log_distance_km"] = np.log1p(data["distance_km"])
    data["sqrt_distance_km"] = np.sqrt(data["distance_km"])
    data["delta_latitude"] = data["dropoff_latitude"] - data["pickup_latitude"]
    data["delta_longitude"] = data["dropoff_longitude"] - data["pickup_longitude"]
    data["abs_delta_latitude"] = data["delta_latitude"].abs()
    data["abs_delta_longitude"] = data["delta_longitude"].abs()
    data["manhattan_km"] = (
        data["abs_delta_latitude"] * KM_PER_DEG_LAT + data["abs_delta_longitude"] * KM_PER_DEG_LON_NYC
    )

    bearing_sin, bearing_cos = _bearing_sin_cos(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )
    data["bearing_sin"] = bearing_sin
    data["bearing_cos"] = bearing_cos

    data["pickup_dist_center"] = haversine(
        data["pickup_latitude"],
        data["pickup_longitude"],
        NYC_MAP_CENTER[0],
        NYC_MAP_CENTER[1],
    )
    data["dropoff_dist_center"] = haversine(
        data["dropoff_latitude"],
        data["dropoff_longitude"],
        NYC_MAP_CENTER[0],
        NYC_MAP_CENTER[1],
    )

    data["distance_x_rush"] = data["distance_km"] * data["is_rush_hour"]
    data["distance_x_weekend"] = data["distance_km"] * data["is_weekend"]

    data["pickup_near_jfk"] = _near_airport(
        data["pickup_latitude"], data["pickup_longitude"], JFK_COORDS[0], JFK_COORDS[1]
    ).astype(int)
    data["dropoff_near_jfk"] = _near_airport(
        data["dropoff_latitude"], data["dropoff_longitude"], JFK_COORDS[0], JFK_COORDS[1]
    ).astype(int)
    data["pickup_near_lga"] = _near_airport(
        data["pickup_latitude"], data["pickup_longitude"], LGA_COORDS[0], LGA_COORDS[1]
    ).astype(int)
    data["dropoff_near_lga"] = _near_airport(
        data["dropoff_latitude"], data["dropoff_longitude"], LGA_COORDS[0], LGA_COORDS[1]
    ).astype(int)
    data["is_short_trip"] = (data["distance_km"] < SHORT_TRIP_KM).astype(int)
    data["is_long_trip"] = (data["distance_km"] > LONG_TRIP_KM).astype(int)
    return data


def engineer_features_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = _add_engineered_columns(df.copy())
    data["vendor_id"] = data["vendor_id"].map({1: 0, 2: 1})
    data["target"] = np.log1p(data["trip_duration"])
    return data


def engineer_inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = _add_engineered_columns(df.copy())
    data["vendor_id"] = data["vendor_id"].map({1: 0, 2: 1})
    return data
