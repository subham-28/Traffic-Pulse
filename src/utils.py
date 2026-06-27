def validate_horizon(horizon):
    allowed = ["5min", "15min", "30min"]

    if horizon not in allowed:
        raise ValueError(f"horizon must be one of {allowed}")

    return horizon