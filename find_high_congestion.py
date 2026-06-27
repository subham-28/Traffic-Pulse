from src.load_artifacts import load_data
from src.predict import predict_from_lat_lon

df = load_data()

if "latitude" not in df.columns and "latitude_x" in df.columns:
    df["latitude"] = df["latitude_x"]

if "longitude" not in df.columns and "longitude_x" in df.columns:
    df["longitude"] = df["longitude_x"]

sensors = df[["sensor_id", "latitude", "longitude"]].drop_duplicates()

results = []

for _, row in sensors.iterrows():
    lat = float(row["latitude"])
    lon = float(row["longitude"])

    try:
        result = predict_from_lat_lon(lat, lon, "5min")
        results.append({
            "sensor_id": row["sensor_id"],
            "latitude": lat,
            "longitude": lon,
            "predicted_speed": result["predicted_speed_mph"],
            "congestion": result["congestion_level"]
        })
    except Exception as e:
        print("Skipped sensor:", row["sensor_id"], "| Error:", e)

results = sorted(results, key=lambda x: x["predicted_speed"])

print("\nTop 10 lowest predicted speed sensors:")
print("-" * 60)

for r in results[:10]:
    print(r)