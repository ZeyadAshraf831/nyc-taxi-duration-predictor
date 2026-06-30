from datetime import datetime

from flask import Flask, jsonify, render_template, request

from taxi_predictor.config import NYC_MAP_CENTER, NYC_MAP_ZOOM
from taxi_predictor.features import build_features, predict_duration
from taxi_predictor.loader import load_artifacts

app = Flask(__name__)
model, scaler, _columns = load_artifacts()


@app.route("/")
def index():
    return render_template(
        "index.html",
        map_lat=NYC_MAP_CENTER[0],
        map_lng=NYC_MAP_CENTER[1],
        map_zoom=NYC_MAP_ZOOM,
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    required = (
        "pickup_lat",
        "pickup_lon",
        "dropoff_lat",
        "dropoff_lon",
        "vendor_id",
        "passenger_count",
        "pickup_datetime",
    )
    missing = [field for field in required if data.get(field) is None]
    if missing:
        return jsonify({"success": False, "error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        pickup_datetime = datetime.fromisoformat(data["pickup_datetime"])
        features, meta = build_features(
            pickup_lat=float(data["pickup_lat"]),
            pickup_lon=float(data["pickup_lon"]),
            dropoff_lat=float(data["dropoff_lat"]),
            dropoff_lon=float(data["dropoff_lon"]),
            vendor_id=int(data["vendor_id"]),
            passenger_count=int(data["passenger_count"]),
            pickup_datetime=pickup_datetime,
            columns=_columns,
        )
        duration = predict_duration(model, scaler, features, columns=_columns)
    except (TypeError, ValueError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    if duration != duration or duration <= 0:
        return jsonify(
            {
                "success": False,
                "error": "The selected coordinates appear to be outside the NYC service area.",
            }
        ), 422

    minutes = int(duration // 60)
    seconds = int(duration % 60)

    return jsonify(
        {
            "success": True,
            "duration_seconds": round(duration),
            "minutes": minutes,
            "seconds": seconds,
            "distance_km": round(meta["distance_km"], 2),
            "is_rush_hour": bool(meta["is_rush_hour"]),
            "is_weekend": bool(meta["is_weekend"]),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
