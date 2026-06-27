import numpy as np


def haversine(lat1, lon1, lat2, lon2):
    radius_km = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * radius_km * np.arcsin(np.sqrt(a))


def find_nearest_sensor(df, lat, lon):
    sensors = df[["sensor_id", "latitude", "longitude"]].drop_duplicates().copy()

    sensors["distance_km"] = haversine(
        lat,
        lon,
        sensors["latitude"],
        sensors["longitude"],
    )

    return sensors.sort_values("distance_km").iloc[0]