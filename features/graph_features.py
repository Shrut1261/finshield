"""Graph-topology features: cross-entity signals from shared infrastructure.

Fraudsters reuse devices and IP addresses across multiple stolen cards.
A device used by 5 distinct customers is a strong ring-fraud indicator.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def compute_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add device/IP sharing counts and merchant fraud rate features.

    Args:
        df: DataFrame with columns [transaction_id, customer_id, card_id,
            device_id, ip_address, merchant_id, is_fraud].

    Returns:
        DataFrame with graph feature columns appended.
    """
    df = df.copy()

    # ── Device sharing ─────────────────────────────────────────────────────────
    if "device_id" in df.columns:
        device_card_count = (
            df.groupby("device_id")["card_id"].nunique().rename("device_id_card_count")
        )
        device_cust_count = (
            df.groupby("device_id")["customer_id"].nunique().rename("device_id_customer_count")
        )
        df = df.merge(device_card_count, on="device_id", how="left")
        df = df.merge(device_cust_count, on="device_id", how="left")
        df["device_id_card_count"] = df["device_id_card_count"].fillna(1).astype("int16")
        df["device_id_customer_count"] = df["device_id_customer_count"].fillna(1).astype("int16")
        df["customer_shared_device_flag"] = (df["device_id_customer_count"] > 1).astype("int8")
    else:
        df["device_id_card_count"] = 1
        df["device_id_customer_count"] = 1
        df["customer_shared_device_flag"] = 0

    # ── IP sharing ─────────────────────────────────────────────────────────────
    if "ip_address" in df.columns:
        ip_card_count = (
            df.groupby("ip_address")["card_id"].nunique().rename("ip_address_card_count")
        )
        df = df.merge(ip_card_count, on="ip_address", how="left")
        df["ip_address_card_count"] = df["ip_address_card_count"].fillna(1).astype("int16")
    else:
        df["ip_address_card_count"] = 1

    # ── Merchant fraud rate (rolling, computed from same DataFrame) ────────────
    if "is_fraud" in df.columns and "merchant_id" in df.columns:
        merchant_fraud_rate = (
            df.groupby("merchant_id")["is_fraud"]
            .mean()
            .rename("merchant_fraud_rate")
            .round(6)
        )
        df = df.merge(merchant_fraud_rate, on="merchant_id", how="left")
        df["merchant_fraud_rate"] = df["merchant_fraud_rate"].fillna(0.0)
    else:
        df["merchant_fraud_rate"] = 0.0

    # ── Card age proxy: first seen vs current transaction ─────────────────────
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        card_first_seen = (
            df.groupby("card_id")["timestamp"].min().rename("card_first_seen")
        )
        df = df.merge(card_first_seen, on="card_id", how="left")
        df["card_age_days"] = (
            (df["timestamp"] - df["card_first_seen"]).dt.total_seconds() / 86_400
        ).round(1)
        df.drop(columns=["card_first_seen"], inplace=True)
    else:
        df["card_age_days"] = 0.0

    logger.debug("Graph features computed for %d transactions", len(df))
    return df
