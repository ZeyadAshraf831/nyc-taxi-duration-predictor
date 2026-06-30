import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from taxi_predictor.config import (
    BASELINE_R2,
    COLUMNS_PATH,
    FEATURE_COLUMNS,
    MODEL_PATH,
    PREVIOUS_DNN_R2,
    SCALER_PATH,
    SELECTED_FEATURES_PATH,
    TRAIN_CSV_PATH,
)
from taxi_predictor.model import build_model
from taxi_predictor.preprocessing import prepare_training_data


def _load_feature_columns() -> list[str]:
    if SELECTED_FEATURES_PATH.exists():
        with open(SELECTED_FEATURES_PATH, "rb") as f:
            return pickle.load(f)
    return list(FEATURE_COLUMNS)


def train(
    train_csv: Path = TRAIN_CSV_PATH,
    epochs: int = 100,
    batch_size: int = 8192,
    learning_rate: float = 8e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    hidden_dim: int = 256,
    num_blocks: int = 4,
    dropout: float = 0.15,
):
    if not train_csv.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_csv}\n"
            "Download train.csv from the NYC Taxi Trip Duration Kaggle competition first."
        )

    feature_columns = _load_feature_columns()
    print(f"Loading data from {train_csv} ...")
    df = pd.read_csv(train_csv)
    x_train, x_test, y_train, y_test, scaler = prepare_training_data(df, feature_columns=feature_columns)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Feature count: {x_train.shape[1]}")
    print(f"Train shape: {x_train.shape} | Validation shape: {x_test.shape}")

    x_train_t = torch.tensor(x_train.values, dtype=torch.float32)
    x_test_t = torch.tensor(x_test.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=batch_size)

    model = build_model(input_dim=x_train.shape[1], legacy=False)
    model = model.to(device)

    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    current_lr = learning_rate

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            print(f"Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}")
            current_lr = new_lr

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {current_lr:.6f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()

    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds.append(model(xb.to(device)).cpu())
            actuals.append(yb)

    preds = torch.cat(preds).numpy()
    actuals = torch.cat(actuals).numpy()
    preds_seconds = np.expm1(preds)
    actuals_seconds = np.expm1(actuals)

    rmse = float(np.sqrt(np.mean((preds_seconds - actuals_seconds) ** 2)))
    r2 = float(1 - np.sum((actuals_seconds - preds_seconds) ** 2) / np.sum((actuals_seconds - actuals_seconds.mean()) ** 2))
    print(f"Validation RMSE: {rmse:.2f} seconds")
    print(f"Validation R2:   {r2:.3f}")
    print(f"Baseline R2:     {BASELINE_R2:.3f}")
    print(f"Previous DNN R2: {PREVIOUS_DNN_R2:.3f}")
    print(f"R2 vs previous:  {r2 - PREVIOUS_DNN_R2:+.3f}")

    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(COLUMNS_PATH, "wb") as f:
        pickle.dump(list(feature_columns), f)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved feature columns to {COLUMNS_PATH}")


if __name__ == "__main__":
    train()
