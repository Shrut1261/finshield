"""Behavioral deviation features: captures 'is this normal for this customer?'

A 3-sigma amount deviation for a customer whose avg ticket is $45 is a strong
fraud signal even if the absolute amount is low.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Rolling window for customer baseline statistics
BASELINE_WINDOW_DAYS = 90


def compute_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add customer behavioral deviation features to the transaction DataFrame.

    Args:
        df: DataFrame with columns [transaction_id, customer_id, amount,
            merchant_category, country, timestamp].

    Returns:
        DataFrame with additional behavioral feature columns.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Per-customer rolling statistics (90-day lookback)
    amount_zscores, amount_ratios, days_since_last = [], [], []
    is_new_category, is_new_country = [], []
    time_since_last_tx_hours = []

    for idx, row in df.iterrows():
        cust = row["customer_id"]
        cutoff = row["timestamp"] - pd.Timedelta(days=BASELINE_WINDOW_DAYS)

        history = df.loc[
            (df["customer_id"] == cust)
            & (df["timestamp"] < row["timestamp"])
            & (df["timestamp"] >= cutoff)
        ]

        if len(history) >= 2:
            hist_mean = history["amount"].mean()
            hist_std = history["amount"].std()
            zscore = (row["amount"] - hist_mean) / (hist_std + 1e-9)
            ratio = row["amount"] / (hist_mean + 1e-9)
        else:
            zscore = 0.0
            ratio = 1.0

        amount_zscores.append(round(float(zscore), 4))
        amount_ratios.append(round(float(ratio), 4))

        # Time since last transaction
        prior = df.loc[
            (df["customer_id"] == cust) & (df["timestamp"] < row["timestamp"])
        ]
        if len(prior) > 0:
            last_ts = prior["timestamp"].max()
            delta_hours = (row["timestamp"] - last_ts).total_seconds() / 3600.0
            days_since_last.append(round(delta_hours / 24.0, 2))
            time_since_last_tx_hours.append(round(float(delta_hours), 2))
        else:
            days_since_last.append(999.0)
            time_since_last_tx_hours.append(999.0)

        # First-time flags
        hist_categories = set(history["merchant_category"].tolist()) if "merchant_category" in df.columns else set()
        hist_countries = set(history["country"].tolist()) if "country" in df.columns else set()

        is_new_category.append(row.get("merchant_category", "") not in hist_categories)
        is_new_country.append(row.get("country", "") not in hist_countries)

    df["amount_zscore"] = amount_zscores
    df["amount_ratio_30d"] = amount_ratios
    df["days_since_last_tx"] = days_since_last
    df["time_since_last_tx_hours"] = time_since_last_tx_hours
    df["is_new_merchant_category"] = is_new_category
    df["is_new_country"] = is_new_country

    # High-amount flag: top 1% of transaction amounts is suspicious
    p99 = df["amount"].quantile(0.99)
    df["is_high_amount"] = (df["amount"] >= p99).astype(int)

    logger.debug("Behavioral features computed for %d transactions", len(df))
    return df
