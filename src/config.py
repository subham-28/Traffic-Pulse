from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "features" / "traffic_weather_road.parquet"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATHS = {
    "5min": MODEL_DIR / "catboost_5min.pkl",
    "15min": MODEL_DIR / "catboost_15min.pkl",
    "30min": MODEL_DIR / "catboost_30min.pkl",
}

ENCODER_PATH = MODEL_DIR / "highway_encoder.pkl"