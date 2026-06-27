import ast
import re
import numpy as np


DROP_COLS = [
    "timestamp",
    "weather_time",
    "sensor_id",
    "osmid",
    "target_speed_5min",
    "target_speed_15min",
    "target_speed_30min",
]


def extract_first_value(value):
    if value is None:
        return np.nan

    if isinstance(value, float) and np.isnan(value):
        return np.nan

    if isinstance(value, list):
        return value[0] if len(value) > 0 else np.nan

    if isinstance(value, str):
        value = value.strip()

        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed[0] if len(parsed) > 0 else np.nan
            except Exception:
                return value

        return value

    return value


def extract_number(value):
    value = extract_first_value(value)

    if value is None:
        return np.nan

    if isinstance(value, float) and np.isnan(value):
        return np.nan

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    match = re.search(r"\d+\.?\d*", str(value))

    if match:
        return float(match.group())

    return np.nan


def clean_highway(value):
    value = extract_first_value(value)

    if value is None:
        return "unknown"

    if isinstance(value, float) and np.isnan(value):
        return "unknown"

    return str(value).strip()


def safe_encode_highway(series, encoder):
    series = series.apply(clean_highway)

    known_classes = set(encoder.classes_)
    fallback_class = encoder.classes_[0]

    series = series.apply(lambda x: x if x in known_classes else fallback_class)

    return encoder.transform(series)


def prepare_input(row, encoder):
    X = row.copy()

    # Fix duplicate latitude/longitude column names created during merge
    rename_cols = {}

    if "latitude_x" in X.columns:
        rename_cols["latitude_x"] = "latitude"

    if "longitude_x" in X.columns:
        rename_cols["longitude_x"] = "longitude"

    X = X.rename(columns=rename_cols)

    # Drop duplicate road/sensor coordinate columns if present
    X = X.drop(
        columns=["latitude_y", "longitude_y"],
        errors="ignore"
    )

    # Drop non-feature columns
    X = X.drop(columns=DROP_COLS, errors="ignore")

    # Encode highway
    if "highway" in X.columns:
        X["highway"] = safe_encode_highway(X["highway"], encoder)

    # Clean numeric road columns
    numeric_road_cols = [
        "lanes",
        "maxspeed",
        "length",
        "distance_to_road",
    ]

    for col in numeric_road_cols:
        if col in X.columns:
            X[col] = X[col].apply(extract_number)

    # Convert oneway boolean to numeric
    if "oneway" in X.columns:
        X["oneway"] = X["oneway"].astype(int)

    # Convert any remaining object columns to numeric
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].apply(extract_number)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X