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

# Pandas offset aliases for time-based rolling
_WINDOW_OFFSETS: dict[str, str] = {
    "1h": "1h",
    "24h": "24h",
    "7d": "7D",
}

# Window durations in seconds (kept for SQL function and unique-count fallback)
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
        Input DataFrame with 22 additional velocity feature columns appended.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Rolling count / sum / mean via pandas groupby-rolling (O(n log n)) ──────
    # closed="left" gives the half-open window [t - W, t), excluding the current tx.
    # We process each entity×window pair by iterating over groups so that we can
    # map results back to the original integer index without timestamp-collision risk.
    for entity_col in ("customer_id", "card_id"):
        for window_label, window_offset in _WINDOW_OFFSETS.items():
            col_prefix = f"{entity_col}_{window_label}"
            counts = np.zeros(len(df), dtype=np.float32)
            sums = np.zeros(len(df), dtype=np.float64)
            means = np.zeros(len(df), dtype=np.float64)

            for _, group in df.groupby(entity_col, sort=False):
                # group rows are in timestamp-ascending order (df is pre-sorted)
                g = group.set_index("timestamp")["amount"]
                rolling = g.rolling(window_offset, closed="left")
                orig = group.index.values  # positions in df
                counts[orig] = rolling.count().values
                sums[orig] = rolling.sum().values
                means[orig] = rolling.mean().fillna(0.0).values

            df[f"tx_count_{col_prefix}"] = counts.astype(int)
            df[f"tx_sum_{col_prefix}"] = np.round(sums, 2)
            df[f"tx_mean_{col_prefix}"] = np.round(means, 2)

    # ── Unique merchants / countries (numpy-vectorized per entity group) ─────────
    # O(k²) per customer where k = txns per customer — fast in practice (k << N).
    has_merchant = "merchant_id" in df.columns
    has_country = "country" in df.columns

    for window_label, seconds in [("24h", 86_400), ("7d", 604_800)]:
        ns = np.int64(seconds) * 1_000_000_000  # nanoseconds
        uniq_merchants = np.zeros(len(df), dtype=np.int32)
        uniq_countries = np.zeros(len(df), dtype=np.int32)

        for _, group in df.groupby("customer_id", sort=False):
            ts = group["timestamp"].values.astype("int64")
            orig = group.index.values
            merchant_vals = group["merchant_id"].values if has_merchant else None
            country_vals = group["country"].values if has_country else None

            for i in range(len(group)):
                in_win = (ts >= ts[i] - ns) & (ts < ts[i])
                if merchant_vals is not None:
                    uniq_merchants[orig[i]] = int(np.unique(merchant_vals[in_win]).size)
                if country_vals is not None:
                    uniq_countries[orig[i]] = int(np.unique(country_vals[in_win]).size)

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
