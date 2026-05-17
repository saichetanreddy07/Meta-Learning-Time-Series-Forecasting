import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


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
            "skewness": np.nan,
            "kurtosis": np.nan,
            "seasonality_strength": np.nan,
            "adf_statistic": np.nan,
        }

    # Mean
    mean = float(np.mean(y))

    # Variance
    variance = float(np.var(y, ddof=1)) if n > 1 else 0.0

    # Standard deviation
    std = float(np.sqrt(variance))

    # Trend calculation
    if n > 1:
        x = np.arange(n, dtype=float)
        slope, _ = np.polyfit(x, y, 1)
        trend = float(slope)
    else:
        trend = 0.0

    # Autocorrelation
    if n > 2:
        autocorr = float(np.corrcoef(y[1:], y[:-1])[0, 1])
    else:
        autocorr = 0.0

    # Skewness
    skewness = float(pd.Series(y).skew()) if n > 2 else 0.0

    # Kurtosis
    kurtosis = float(pd.Series(y).kurt()) if n > 3 else 0.0

    # Seasonality strength approximation
    if n > 12:
        seasonal_diff = np.mean(np.abs(y[12:] - y[:-12]))
        seasonality_strength = float(seasonal_diff)
    else:
        seasonality_strength = 0.0

    # ADF Statistic
    try:
        adf_statistic = float(adfuller(y)[0])
    except Exception:
        adf_statistic = 0.0

    return {
        "length": int(n),
        "mean": mean,
        "variance": variance,
        "standard_deviation": std,
        "trend": trend,
        "autocorrelation": autocorr,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "seasonality_strength": seasonality_strength,
        "adf_statistic": adf_statistic,
    }