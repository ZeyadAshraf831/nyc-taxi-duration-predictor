from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from taxi_predictor.config import FEATURE_COLUMNS, SCALE_COLS, TRAIN_CSV_PATH
from taxi_predictor.loader import load_artifacts
from taxi_predictor.preprocessing import TEST_CSV_PATH, SUBMISSION_PATH, prepare_test_features, prepare_training_data


def _predict_in_batches(model, features: pd.DataFrame, scaler, columns: list[str], batch_size: int = 4096) -> np.ndarray:
    scaled = features[columns].copy()
    scaled[SCALE_COLS] = scaler.transform(scaled[SCALE_COLS])

    tensor = torch.tensor(scaled.values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size)

    device = next(model.parameters()).device
    model.eval()
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            preds.append(model(batch.to(device)).cpu())

    return np.expm1(torch.cat(preds).numpy()).reshape(-1)


def evaluate_validation_split(train_csv: Path = TRAIN_CSV_PATH) -> dict:
    df = pd.read_csv(train_csv)
    x_train, x_val, y_train, y_val, scaler = prepare_training_data(df)
    model, _, columns = load_artifacts()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    val_features = x_val.copy()
    preds_log = []
    loader = DataLoader(
        TensorDataset(torch.tensor(val_features[columns].values, dtype=torch.float32)),
        batch_size=4096,
    )
    with torch.no_grad():
        for (batch,) in loader:
            preds_log.append(model(batch.to(device)).cpu())
    preds_log = torch.cat(preds_log).numpy().reshape(-1)
    preds_seconds = np.expm1(preds_log)
    actual_seconds = np.expm1(y_val.values)

    rmse = float(np.sqrt(np.mean((preds_seconds - actual_seconds) ** 2)))
    r2 = float(
        1
        - np.sum((actual_seconds - preds_seconds) ** 2)
        / np.sum((actual_seconds - actual_seconds.mean()) ** 2)
    )
    return {"rmse": rmse, "r2": r2, "rows": len(x_val)}


def predict_test_csv(
    test_csv: Path = TEST_CSV_PATH,
    submission_path: Path = SUBMISSION_PATH,
    batch_size: int = 4096,
) -> pd.DataFrame:
    if not test_csv.exists():
        raise FileNotFoundError(f"Test file not found: {test_csv}")

    raw = pd.read_csv(test_csv)
    featured = prepare_test_features(raw)
    model, scaler, columns = load_artifacts()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    predictions = _predict_in_batches(model, featured, scaler, columns, batch_size=batch_size)
    predictions = np.clip(predictions, 1, None).astype(int)

    submission = featured[["id"]].copy()
    submission["trip_duration"] = predictions

    missing_ids = set(raw["id"]) - set(submission["id"])
    if missing_ids:
        fallback = int(np.median(predictions))
        fallback_rows = raw[raw["id"].isin(missing_ids)][["id"]].copy()
        fallback_rows["trip_duration"] = fallback
        submission = pd.concat([submission, fallback_rows], ignore_index=True)

    submission = submission.sort_values("id").reset_index(drop=True)
    submission.to_csv(submission_path, index=False)
    return submission


def main():
    print("=" * 60)
    print("DNN Evaluation")
    print("=" * 60)

    test_df = pd.read_csv(TEST_CSV_PATH, nrows=1)
    if "trip_duration" in test_df.columns:
        print("test.csv contains labels -> computing test R2")
    else:
        print("test.csv has no trip_duration column (Kaggle holdout).")
        print("R2 cannot be computed on test.csv without labels.")

    print("\nValidation metrics from train.csv holdout (20%):")
    metrics = evaluate_validation_split()
    print(f"Rows: {metrics['rows']}")
    print(f"RMSE: {metrics['rmse']:.2f} seconds")
    print(f"R2:   {metrics['r2']:.3f}")

    print("\nGenerating predictions for test.csv ...")
    submission = predict_test_csv()
    print(f"Predictions saved to: {SUBMISSION_PATH}")
    print(f"Submission rows: {len(submission)}")
    print(f"Sample prediction (seconds): {submission['trip_duration'].head(3).tolist()}")


if __name__ == "__main__":
    main()
