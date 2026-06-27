from src.load_artifacts import load_data, load_encoder, load_model
from src.nearest_sensor import find_nearest_sensor
from src.preprocessing import prepare_input


def get_congestion_level(speed):
    if speed >= 50:
        return "Low"
    elif speed >= 30:
        return "Medium"
    return "High"


def predict_from_lat_lon(lat, lon, horizon="5min"):
    df = load_data()
    model = load_model(horizon)
    encoder = load_encoder()

    # Create separate sensor dataframe for nearest sensor search
    sensor_df = (
        df[["sensor_id", "latitude_x", "longitude_x"]]
        .drop_duplicates()
        .rename(columns={
            "latitude_x": "latitude",
            "longitude_x": "longitude"
        })
        .copy()
    )

    sensor_df["sensor_id"] = sensor_df["sensor_id"].astype(str)
    df["sensor_id"] = df["sensor_id"].astype(str)

    nearest_sensor = find_nearest_sensor(sensor_df, lat, lon)
    sensor_id = nearest_sensor["sensor_id"]

    latest_row = (
        df[df["sensor_id"] == sensor_id]
        .sort_values("timestamp")
        .tail(1)
    )

    if latest_row.empty:
        raise ValueError(f"No data found for nearest sensor: {sensor_id}")

    X = prepare_input(latest_row, encoder)

    # Match CatBoost training feature names and order exactly
    expected_features = model.feature_names_
    
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0
    
    X = X[expected_features]
    
    # print("Final prediction input columns:")
    # print(X.columns.tolist())
    
    # print("\nFinal prediction input:")
    # print(X.head())
    
    predicted_speed = float(model.predict(X)[0])

    return {
        "input_latitude": lat,
        "input_longitude": lon,
        "horizon": horizon,
        "nearest_sensor_id": sensor_id,
        "distance_km": round(float(nearest_sensor["distance_km"]), 3),
        "predicted_speed_mph": round(predicted_speed, 2),
        "congestion_level": get_congestion_level(predicted_speed),
    }