import numpy as np
import pandas as pd


def extract_meta_features(series: pd.Series) -> dict:
    y = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    n = len(y)

    if n == 0:
        return {
            "length": 0,
            "mean": np.nan,
            "variance": np.nan,
            "standard_deviation": np.nan,
            "trend": np.nan,
            "autocorrelation": np.nan,
        }

    mean = float(np.mean(y))
    variance = float(np.var(y, ddof=1)) if n > 1 else 0.0
    std = float(np.sqrt(variance))

    if n > 1:
        x = np.arange(n, dtype=float)
        slope, _ = np.polyfit(x, y, 1)
        trend = float(slope)
    else:
        trend = 0.0

    if n > 2:
        autocorr = float(np.corrcoef(y[1:], y[:-1])[0, 1])
    else:
        autocorr = 0.0

    return {
        "length": int(n),
        "mean": mean,
        "variance": variance,
        "standard_deviation": std,
        "trend": trend,
        "autocorrelation": autocorr,
    }