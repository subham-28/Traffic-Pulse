import joblib
import pandas as pd

from src.config import DATA_PATH, MODEL_PATHS, ENCODER_PATH


def load_data():
    return pd.read_parquet(DATA_PATH)


def load_model(horizon="5min"):
    if horizon not in MODEL_PATHS:
        raise ValueError("horizon must be one of: 5min, 15min, 30min")
    return joblib.load(MODEL_PATHS[horizon])


def load_encoder():
    return joblib.load(ENCODER_PATH)