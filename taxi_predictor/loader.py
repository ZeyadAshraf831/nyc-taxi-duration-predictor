import pickle

import joblib
import torch

from taxi_predictor.config import COLUMNS_PATH, MODEL_PATH, SCALER_PATH
from taxi_predictor.model import build_model


def load_artifacts():
    with open(COLUMNS_PATH, "rb") as f:
        columns = pickle.load(f)

    scaler = joblib.load(SCALER_PATH)
    legacy = len(columns) == 13
    model = build_model(input_dim=len(columns), legacy=legacy)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    return model, scaler, columns
