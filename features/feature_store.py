"""SQL-backed versioned feature store.

Persists computed feature vectors to features.feature_vectors as JSONB.
Versioning allows changing feature engineering logic without breaking
production scoring — the API retrieves features by (transaction_id, version).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)

CURRENT_FEATURE_VERSION = "v1"


class FeatureStore:
    """Read/write interface for the SQL feature store."""

    def __init__(self, engine: Engine, version: str = CURRENT_FEATURE_VERSION) -> None:
        self.engine = engine
        self.version = version

    def write(self, df: pd.DataFrame, feature_cols: list[str]) -> int:
        """Persist feature vectors for a batch of transactions.

        Args:
            df: DataFrame containing transaction_id and feature columns.
            feature_cols: Subset of columns to store as the feature vector.

        Returns:
            Number of rows written.
        """
        rows = []
        now = datetime.utcnow()
        for _, row in df.iterrows():
            features_dict = {col: _safe_json(row.get(col)) for col in feature_cols if col in df.columns}
            rows.append(
                {
                    "transaction_id": row["transaction_id"],
                    "feature_version": self.version,
                    "computed_at": now,
                    "features": json.dumps(features_dict),
                }
            )

        if not rows:
            return 0

        insert_sql = text("""
            INSERT INTO features.feature_vectors
                (transaction_id, feature_version, computed_at, features)
            VALUES
                (:transaction_id, :feature_version, :computed_at, :features::jsonb)
            ON CONFLICT (transaction_id)
            DO UPDATE SET
                feature_version = EXCLUDED.feature_version,
                computed_at     = EXCLUDED.computed_at,
                features        = EXCLUDED.features
        """)

        with self.engine.begin() as conn:
            conn.execute(insert_sql, rows)

        logger.info("Wrote %d feature vectors (version=%s)", len(rows), self.version)
        return len(rows)

    def read(self, transaction_ids: list[str]) -> pd.DataFrame:
        """Retrieve feature vectors for a list of transaction IDs."""
        query = text("""
            SELECT transaction_id, feature_version, computed_at, features
            FROM features.feature_vectors
            WHERE transaction_id = ANY(:tx_ids)
              AND feature_version = :version
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {"tx_ids": transaction_ids, "version": self.version})
            rows = result.fetchall()

        if not rows:
            return pd.DataFrame()

        records = []
        for row in rows:
            record = {"transaction_id": row.transaction_id, "feature_version": row.feature_version}
            record.update(row.features if isinstance(row.features, dict) else json.loads(row.features))
            records.append(record)

        return pd.DataFrame(records)

    def read_one(self, transaction_id: str) -> Optional[dict]:
        """Retrieve a single transaction's feature vector as a dict."""
        df = self.read([transaction_id])
        if df.empty:
            return None
        return df.iloc[0].to_dict()


def _safe_json(value: object) -> object:
    """Convert numpy/pandas scalars to JSON-serializable Python types."""
    import numpy as np
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return None
    return value
