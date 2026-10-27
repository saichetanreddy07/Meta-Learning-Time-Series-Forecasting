import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def extract_meta_features(series: pd.Series) -> dict:
    """
    Extract statistical meta-features from a time series.
    """

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

    # Basic statistics
    mean = float(np.mean(y))
    variance = float(np.var(y, ddof=1)) if n > 1 else 0.0
    standard_deviation = float(np.sqrt(variance))

    # Trend (slope of best-fit line)
    if n > 1:
        x = np.arange(n)
        trend = float(np.polyfit(x, y, 1)[0])
    else:
        trend = 0.0

    # Lag-1 autocorrelation
    if n > 2 and np.std(y[:-1]) > 0 and np.std(y[1:]) > 0:
        autocorrelation = float(np.corrcoef(y[:-1], y[1:])[0, 1])
    else:
        autocorrelation = 0.0

    # Distribution statistics
    skewness = float(pd.Series(y).skew()) if n > 2 else 0.0
    kurtosis = float(pd.Series(y).kurt()) if n > 3 else 0.0

    # Simple seasonality approximation
    if n > 12:
        seasonality_strength = float(np.mean(np.abs(y[12:] - y[:-12])))
    else:
        seasonality_strength = 0.0

    # Stationarity (ADF Test)
    try:
        adf_statistic = float(
            adfuller(
                y,
                result_object=True,
            ).statistic
        )
    except Exception:
        adf_statistic = 0.0

    return {
        "length": n,
        "mean": mean,
        "variance": variance,
        "standard_deviation": standard_deviation,
        "trend": trend,
        "autocorrelation": autocorrelation,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "seasonality_strength": seasonality_strength,
        "adf_statistic": adf_statistic,
    }