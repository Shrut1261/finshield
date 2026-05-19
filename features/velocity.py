"""Velocity features: rolling time-window aggregations per customer and card.

These capture rate-of-change signals that indicate card testing, account
takeover bursts, and other velocity-based fraud patterns.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)

# Window durations in seconds
WINDOWS: dict[str, int] = {
    "1h": 3_600,
    "24h": 86_400,
    "7d": 604_800,
}


def compute_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling count/sum/mean per customer and card over 1h, 24h, 7d windows.

    Operates in-memory on a sorted DataFrame. For production at scale, use
    ``compute_velocity_sql`` which pushes the computation to the database.

    Args:
        df: DataFrame with columns [transaction_id, customer_id, card_id,
            amount, timestamp]. Must be sorted by timestamp ascending.

    Returns:
        Input DataFrame with 18 additional velocity feature columns appended.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    for entity_col in ("customer_id", "card_id"):
        for window_label, seconds in WINDOWS.items():
            col_prefix = f"{entity_col}_{window_label}"
            counts, sums, means = [], [], []

            for _, row in df.iterrows():
                cutoff = row["timestamp"] - pd.Timedelta(seconds=seconds)
                mask = (
                    (df[entity_col] == row[entity_col])
                    & (df["timestamp"] >= cutoff)
                    & (df["timestamp"] < row["timestamp"])
                )
                window_amounts = df.loc[mask, "amount"]
                counts.append(len(window_amounts))
                sums.append(float(window_amounts.sum()))
                means.append(float(window_amounts.mean()) if len(window_amounts) > 0 else 0.0)

            df[f"tx_count_{col_prefix}"] = counts
            df[f"tx_sum_{col_prefix}"] = np.round(sums, 2)
            df[f"tx_mean_{col_prefix}"] = np.round(means, 2)

    # Unique merchants and countries per customer in 24h / 7d
    for window_label, seconds in [("24h", 86_400), ("7d", 604_800)]:
        uniq_merchants, uniq_countries = [], []
        for _, row in df.iterrows():
            cutoff = row["timestamp"] - pd.Timedelta(seconds=seconds)
            mask = (
                (df["customer_id"] == row["customer_id"])
                & (df["timestamp"] >= cutoff)
                & (df["timestamp"] < row["timestamp"])
            )
            window = df.loc[mask]
            uniq_merchants.append(window["merchant_id"].nunique() if "merchant_id" in df.columns else 0)
            uniq_countries.append(window["country"].nunique() if "country" in df.columns else 0)
        df[f"unique_merchants_{window_label}"] = uniq_merchants
        df[f"unique_countries_{window_label}"] = uniq_countries

    logger.debug("Velocity features computed: %d columns added", 18 + 4)
    return df


def compute_velocity_sql(engine: Engine, transaction_id: str) -> dict[str, float]:
    """Compute velocity features via SQL for real-time single-transaction scoring.

    More efficient than the in-memory version for production use.
    """
    query = text("""
        WITH tx AS (
            SELECT customer_key, card_key, amount, ingested_at
            FROM warehouse.fact_transactions
            WHERE transaction_id = :tx_id
        )
        SELECT
            -- Customer velocity
            COUNT(f.*) FILTER (WHERE f.ingested_at >= tx.ingested_at - INTERVAL '1 hour')   AS tx_count_customer_1h,
            COUNT(f.*) FILTER (WHERE f.ingested_at >= tx.ingested_at - INTERVAL '24 hours')  AS tx_count_customer_24h,
            COUNT(f.*) FILTER (WHERE f.ingested_at >= tx.ingested_at - INTERVAL '7 days')    AS tx_count_customer_7d,
            COALESCE(SUM(f.amount) FILTER (WHERE f.ingested_at >= tx.ingested_at - INTERVAL '1 hour'),   0) AS tx_sum_customer_1h,
            COALESCE(SUM(f.amount) FILTER (WHERE f.ingested_at >= tx.ingested_at - INTERVAL '24 hours'), 0) AS tx_sum_customer_24h,
            COALESCE(SUM(f.amount) FILTER (WHERE f.ingested_at >= tx.ingested_at - INTERVAL '7 days'),   0) AS tx_sum_customer_7d
        FROM warehouse.fact_transactions f, tx
        WHERE f.customer_key = tx.customer_key
          AND f.ingested_at < tx.ingested_at
          AND f.transaction_id != :tx_id
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"tx_id": transaction_id}).fetchone()
    return dict(row._mapping) if row else {}
