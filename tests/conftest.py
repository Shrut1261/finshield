"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_transactions() -> pd.DataFrame:
    """50 synthetic transactions: 48 legit + 2 fraud."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "transaction_id": [f"TX_{i:04d}" for i in range(n)],
        "customer_id": [f"CUST_{i % 10:04d}" for i in range(n)],
        "card_id": [f"CARD_{i % 15:04d}" for i in range(n)],
        "merchant_id": [f"MERCH_{i % 8:04d}" for i in range(n)],
        "merchant_category": rng.choice(["grocery", "online_retail", "wire_transfer"], n).tolist(),
        "amount": rng.uniform(10.0, 500.0, n).round(2).tolist(),
        "currency": ["USD"] * n,
        "country": rng.choice(["US", "GB", "DE"], n).tolist(),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="10min").tolist(),
        "is_fraud": [0] * 48 + [1, 1],
        "source_dataset": ["creditcard_ulb"] * n,
        "device_id": [f"DEV_{i % 20:04d}" for i in range(n)],
        "ip_address": [f"192.168.1.{i % 50}" for i in range(n)],
    })


@pytest.fixture()
def feature_matrix(sample_transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Quick feature set without slow velocity lookbacks."""
    from features.time_features import compute_time_features
    from features.graph_features import compute_graph_features

    df = compute_time_features(sample_transactions, timestamp_col="timestamp")
    df = compute_graph_features(df)

    feature_cols = [
        "amount", "hour_of_day", "day_of_week", "is_weekend",
        "device_id_card_count", "merchant_fraud_rate",
    ]
    X = df[feature_cols].fillna(0).astype("float32")
    y = df["is_fraud"].astype(int)
    return X, y
