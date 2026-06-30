from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COLUMNS_PATH = PROJECT_ROOT / "columns.pkl"
SCALER_PATH = PROJECT_ROOT / "scaler.pkl"
MODEL_PATH = PROJECT_ROOT / "taxi_model.pth"
SELECTED_FEATURES_PATH = PROJECT_ROOT / "selected_features.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"

RUSH_HOURS = [7, 8, 9, 17, 18, 19]
NYC_MAP_CENTER = (40.7484, -73.9967)
NYC_MAP_ZOOM = 12
NYC_MAP_TILES = "OpenStreetMap"

JFK_COORDS = (40.6413, -73.7781)
LGA_COORDS = (40.7769, -73.8740)
AIRPORT_RADIUS_KM = 2.5
SHORT_TRIP_KM = 1.0
LONG_TRIP_KM = 15.0

LEGACY_FEATURE_COLUMNS = [
    "vendor_id",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "distance_km",
    "pickup_hour",
    "pickup_day",
    "pickup_month",
    "pickup_weekday",
    "is_rush_hour",
    "is_weekend",
]

FEATURE_COLUMNS = [
    "vendor_id",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "distance_km",
    "log_distance_km",
    "sqrt_distance_km",
    "manhattan_km",
    "delta_latitude",
    "delta_longitude",
    "abs_delta_latitude",
    "abs_delta_longitude",
    "bearing_sin",
    "bearing_cos",
    "pickup_dist_center",
    "dropoff_dist_center",
    "pickup_near_jfk",
    "dropoff_near_jfk",
    "pickup_near_lga",
    "dropoff_near_lga",
    "is_short_trip",
    "is_long_trip",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "pickup_day",
    "distance_x_rush",
    "distance_x_weekend",
    "is_rush_hour",
    "is_weekend",
    "is_night",
    "is_morning_rush",
    "is_evening_rush",
]

BINARY_FEATURES = {
    "vendor_id",
    "pickup_near_jfk",
    "dropoff_near_jfk",
    "pickup_near_lga",
    "dropoff_near_lga",
    "is_short_trip",
    "is_long_trip",
    "is_rush_hour",
    "is_weekend",
    "is_night",
    "is_morning_rush",
    "is_evening_rush",
}

LEGACY_SCALE_COLS = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "distance_km",
    "passenger_count",
    "pickup_hour",
    "pickup_day",
    "pickup_month",
    "pickup_weekday",
]

SCALE_COLS = [col for col in FEATURE_COLUMNS if col not in BINARY_FEATURES]

TRAIN_CSV_PATH = PROJECT_ROOT / "train.csv"

BASELINE_R2 = 0.790
PREVIOUS_DNN_R2 = 0.799
IMPORTANCE_THRESHOLD = 0.0005

MUST_KEEP_FEATURES = {
    "vendor_id",
    "pickup_near_jfk",
    "dropoff_near_jfk",
    "pickup_near_lga",
    "dropoff_near_lga",
    "is_short_trip",
    "is_long_trip",
    "distance_km",
    "manhattan_km",
    "bearing_sin",
    "bearing_cos",
}
