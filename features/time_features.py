"""Temporal and cyclical time features.

Raw hour (0–23) and day-of-week (0–6) are ordinal but not cyclical — a model
treating hour=23 and hour=0 as far apart would be wrong.  Sin/cos encoding
wraps the cycle correctly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add temporal and cyclical time features.

    Args:
        df: DataFrame containing a timestamp column.
        timestamp_col: Name of the datetime column.

    Returns:
        DataFrame with time feature columns appended.
    """
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])

    df["hour_of_day"] = ts.dt.hour.astype("int8")
    df["day_of_week"] = ts.dt.dayofweek.astype("int8")
    df["month"] = ts.dt.month.astype("int8")
    df["week_of_year"] = ts.dt.isocalendar().week.astype("int16")
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")
    df["is_business_hours"] = (
        ts.dt.dayofweek.between(0, 4) & ts.dt.hour.between(9, 17)
    ).astype("int8")

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24.0).round(6)
    df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24.0).round(6)
    df["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0).round(6)
    df["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0).round(6)
    df["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12.0).round(6)
    df["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12.0).round(6)

    # US federal holiday flag (simplified: major holidays only)
    US_HOLIDAYS = {
        (1, 1), (7, 4), (12, 25), (11, 11),  # New Year, July 4, Christmas, Veterans
    }
    df["is_us_holiday"] = ts.apply(
        lambda t: int((t.month, t.day) in US_HOLIDAYS)
    ).astype("int8")

    return df
