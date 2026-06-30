import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from taxi_predictor.config import (
    FEATURE_COLUMNS,
    IMPORTANCE_THRESHOLD,
    MUST_KEEP_FEATURES,
    REPORTS_DIR,
    SCALE_COLS,
    SELECTED_FEATURES_PATH,
    TRAIN_CSV_PATH,
)
from taxi_predictor.loader import load_artifacts
from taxi_predictor.preprocessing import prepare_training_data


def _predict_seconds(model, features: pd.DataFrame, columns: list[str]) -> np.ndarray:
    device = next(model.parameters()).device
    tensor = torch.tensor(features[columns].values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=4096)
    preds = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            preds.append(model(batch.to(device)).cpu())
    return np.expm1(torch.cat(preds).numpy()).reshape(-1)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_validation_frame(train_csv: Path = TRAIN_CSV_PATH, sample_size: int | None = None):
    df = pd.read_csv(train_csv)
    _, x_val, _, y_val, scaler = prepare_training_data(df, feature_columns=FEATURE_COLUMNS)
    actual_seconds = np.expm1(y_val.values)

    if sample_size and len(x_val) > sample_size:
        idx = np.random.default_rng(42).choice(len(x_val), size=sample_size, replace=False)
        x_val = x_val.iloc[idx].reset_index(drop=True)
        actual_seconds = actual_seconds[idx]

    return x_val, actual_seconds, scaler


def plot_residual_analysis(
    residuals: np.ndarray,
    actual_seconds: np.ndarray,
    distance_km: np.ndarray,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    summary = {}

    # Error distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=80, color="#2563eb", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#dc2626", linestyle="--", linewidth=1.5)
    ax.set_title("Residual Distribution (Predicted - Actual)")
    ax.set_xlabel("Error (seconds)")
    ax.set_ylabel("Trip count")
    fig.tight_layout()
    fig.savefig(reports_dir / "error_distribution.png", dpi=150)
    plt.close(fig)

    # Residual vs distance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(distance_km, residuals, s=4, alpha=0.15, color="#0f766e")
    ax.axhline(0, color="#dc2626", linestyle="--", linewidth=1.2)
    ax.set_title("Residuals vs Trip Distance")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Error (seconds)")
    fig.tight_layout()
    fig.savefig(reports_dir / "residual_vs_distance.png", dpi=150)
    plt.close(fig)

    # Residual vs actual duration
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(actual_seconds, residuals, s=4, alpha=0.15, color="#7c3aed")
    ax.axhline(0, color="#dc2626", linestyle="--", linewidth=1.2)
    ax.set_title("Residuals vs Actual Duration")
    ax.set_xlabel("Actual duration (seconds)")
    ax.set_ylabel("Error (seconds)")
    fig.tight_layout()
    fig.savefig(reports_dir / "residual_vs_actual.png", dpi=150)
    plt.close(fig)

    short_mask = distance_km < 1.0
    medium_mask = (distance_km >= 1.0) & (distance_km <= 5.0)
    long_mask = distance_km > 5.0

    buckets = {
        "short_<1km": residuals[short_mask],
        "medium_1_5km": residuals[medium_mask],
        "long_>5km": residuals[long_mask],
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        [values for values in buckets.values() if len(values)],
        tick_labels=[name for name, values in buckets.items() if len(values)],
        patch_artist=True,
    )
    ax.axhline(0, color="#dc2626", linestyle="--", linewidth=1.2)
    ax.set_title("Residuals by Distance Bucket")
    ax.set_ylabel("Error (seconds)")
    fig.tight_layout()
    fig.savefig(reports_dir / "residual_by_distance_bucket.png", dpi=150)
    plt.close(fig)

    for name, values in buckets.items():
        if len(values):
            summary[name] = {
                "count": int(len(values)),
                "mean_error": float(values.mean()),
                "mae": float(np.abs(values).mean()),
            }

    return summary


def permutation_importance(
    model,
    x_val: pd.DataFrame,
    actual_seconds: np.ndarray,
    columns: list[str],
    repeats: int = 2,
) -> pd.DataFrame:
    baseline_pred = _predict_seconds(model, x_val, columns)
    baseline_rmse = _rmse(actual_seconds, baseline_pred)

    importances = []
    rng = np.random.default_rng(42)

    for col in columns:
        losses = []
        for _ in range(repeats):
            shuffled = x_val.copy()
            shuffled[col] = rng.permutation(shuffled[col].values)
            pred = _predict_seconds(model, shuffled, columns)
            losses.append(_rmse(actual_seconds, pred))

        mean_loss = float(np.mean(losses))
        importances.append(
            {
                "feature": col,
                "rmse_increase": mean_loss - baseline_rmse,
                "importance_pct": (mean_loss - baseline_rmse) / baseline_rmse * 100,
            }
        )

    importance_df = pd.DataFrame(importances).sort_values("rmse_increase", ascending=False)
    return importance_df


def select_features(importance_df: pd.DataFrame, threshold: float = IMPORTANCE_THRESHOLD) -> list[str]:
    selected = set(importance_df[importance_df["rmse_increase"] >= threshold]["feature"].tolist())
    selected.update(MUST_KEEP_FEATURES)
    ordered = [col for col in FEATURE_COLUMNS if col in selected]
    return ordered


def run_analysis(sample_size: int = 20000, save_selected: bool = True) -> dict:
    x_val, actual_seconds, _ = load_validation_frame(sample_size=sample_size)
    model, _, columns = load_artifacts()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if columns != FEATURE_COLUMNS:
        print("Warning: loaded model feature count differs from current FEATURE_COLUMNS.")
        print(f"Model columns: {len(columns)} | Config columns: {len(FEATURE_COLUMNS)}")

    use_columns = [col for col in columns if col in x_val.columns]
    preds = _predict_seconds(model, x_val, use_columns)
    residuals = preds - actual_seconds
    distance_km = x_val["distance_km"].values

    residual_summary = plot_residual_analysis(residuals, actual_seconds, distance_km)
    importance_df = permutation_importance(model, x_val, actual_seconds, use_columns)
    selected = select_features(importance_df)

    importance_path = REPORTS_DIR / "feature_importance.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_path, index=False)

    if save_selected:
        with open(SELECTED_FEATURES_PATH, "wb") as f:
            pickle.dump(selected, f)

    rmse = _rmse(actual_seconds, preds)
    r2 = float(1 - np.sum(residuals**2) / np.sum((actual_seconds - actual_seconds.mean()) ** 2))

    return {
        "rmse": rmse,
        "r2": r2,
        "residual_summary": residual_summary,
        "selected_features": selected,
        "dropped_features": [col for col in FEATURE_COLUMNS if col not in selected],
        "importance_path": str(importance_path),
        "reports_dir": str(REPORTS_DIR),
    }
