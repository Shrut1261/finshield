"""Feature engineering pipeline: orchestrates all feature modules.

Usage:
    pipeline = FeaturePipeline()
    df_with_features = pipeline.run(df_raw)
    feature_cols = pipeline.feature_columns
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import Engine

from features.behavioral import compute_behavioral_features
from features.feature_store import FeatureStore
from features.graph_features import compute_graph_features
from features.time_features import compute_time_features
from features.velocity import compute_velocity_features

logger = logging.getLogger(__name__)

# Canonical ordered feature list for model training
FEATURE_COLUMNS: list[str] = [
    # Velocity — customer
    "tx_count_customer_id_1h", "tx_count_customer_id_24h", "tx_count_customer_id_7d",
    "tx_sum_customer_id_1h", "tx_sum_customer_id_24h", "tx_sum_customer_id_7d",
    "tx_mean_customer_id_1h", "tx_mean_customer_id_24h", "tx_mean_customer_id_7d",
    # Velocity — card
    "tx_count_card_id_1h", "tx_count_card_id_24h", "tx_count_card_id_7d",
    "tx_sum_card_id_1h", "tx_sum_card_id_24h", "tx_sum_card_id_7d",
    "tx_mean_card_id_1h", "tx_mean_card_id_24h", "tx_mean_card_id_7d",
    # Velocity — diversity
    "unique_merchants_24h", "unique_countries_24h",
    "unique_merchants_7d", "unique_countries_7d",
    # Behavioral
    "amount_zscore", "amount_ratio_30d",
    "days_since_last_tx", "time_since_last_tx_hours",
    "is_new_merchant_category", "is_new_country", "is_high_amount",
    # Time
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "is_business_hours", "is_us_holiday",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    # Graph
    "device_id_card_count", "device_id_customer_count",
    "customer_shared_device_flag", "ip_address_card_count",
    "merchant_fraud_rate", "card_age_days",
    # Raw
    "amount",
]


class FeaturePipeline:
    """Runs all feature engineering steps in the correct order and persists results."""

    def __init__(
        self,
        engine: Optional[Engine] = None,
        persist_to_store: bool = False,
        feature_version: str = "v1",
    ) -> None:
        self.engine = engine
        self.persist_to_store = persist_to_store and engine is not None
        self.feature_store = FeatureStore(engine, version=feature_version) if engine else None
        self.feature_columns = FEATURE_COLUMNS

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature modules and return a DataFrame ready for modeling.

        Steps run in order:
          1. Time features (fast, no lookbacks required)
          2. Graph features (aggregate operations within the batch)
          3. Behavioral features (per-customer lookbacks)
          4. Velocity features (rolling windows — slowest step)
          5. Optional: persist to feature store
        """
        logger.info("Starting feature pipeline on %d transactions", len(df))

        df = compute_time_features(df, timestamp_col="timestamp" if "timestamp" in df.columns else "raw_timestamp")
        logger.info("Time features done")

        df = compute_graph_features(df)
        logger.info("Graph features done")

        df = compute_behavioral_features(df)
        logger.info("Behavioral features done")

        df = compute_velocity_features(df)
        logger.info("Velocity features done")

        if self.persist_to_store and self.feature_store:
            written = self.feature_store.write(df, self.feature_columns)
            logger.info("Persisted %d feature vectors to store", written)

        present_cols = [c for c in self.feature_columns if c in df.columns]
        missing_cols = [c for c in self.feature_columns if c not in df.columns]
        if missing_cols:
            logger.warning("Missing feature columns (will be filled with 0): %s", missing_cols)
            for col in missing_cols:
                df[col] = 0

        return df

    def get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the model input columns, filling any gaps with 0."""
        df = self.run(df)
        return df[self.feature_columns].fillna(0).astype("float32")
