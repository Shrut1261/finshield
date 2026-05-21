"""PSI-based model drift monitor.

Population Stability Index measures how much the distribution of each
input feature has shifted from training time to current production.

  PSI < 0.10  → Stable (no action)
  PSI 0.10–0.20 → Monitor (schedule review)
  PSI > 0.20  → Retrain alert

Run as a weekly Azure Function (timer trigger) or via cron.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)

PSI_STABLE_THRESHOLD = 0.10
PSI_MONITOR_THRESHOLD = 0.20
N_BINS = 10


def _psi_score(expected: np.ndarray, actual: np.ndarray, bins: int = N_BINS) -> float:
    """Compute Population Stability Index between two distributions."""
    # Create bins from combined range
    combined = np.concatenate([expected, actual])
    bin_edges = np.percentile(combined, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)  # Remove duplicates from degenerate distributions

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Add epsilon to avoid log(0)
    eps = 1e-6
    expected_pct = expected_counts / max(expected_counts.sum(), 1) + eps
    actual_pct = actual_counts / max(actual_counts.sum(), 1) + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def _psi_status(psi: float) -> str:
    if psi < PSI_STABLE_THRESHOLD:
        return "stable"
    if psi < PSI_MONITOR_THRESHOLD:
        return "monitor"
    return "retrain"


class DriftMonitor:
    """Computes and persists feature-level PSI drift scores.

    Args:
        engine: SQLAlchemy engine connected to the FinShield database.
        model_version: Model version string to tag drift records with.
        training_data_path: Optional path to parquet file with training
            feature distributions (baseline). If None, uses the oldest
            week of production data as baseline.
    """

    def __init__(
        self,
        engine: Engine,
        model_version: str = "v1",
        training_data_path: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.model_version = model_version
        self.training_data_path = training_data_path

    def _load_baseline(self) -> pd.DataFrame:
        if self.training_data_path:
            return pd.read_parquet(self.training_data_path)

        # Fall back to the first full week in the feature store as baseline
        query = text("""
            SELECT features
            FROM features.feature_vectors
            WHERE computed_at < NOW() - INTERVAL '14 days'
            LIMIT 50000
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(query).fetchall()

        if not rows:
            raise RuntimeError("No baseline data available for drift monitoring.")

        import json
        records = [
            r.features if isinstance(r.features, dict) else json.loads(r.features)
            for r in rows
        ]
        return pd.DataFrame(records)

    def _load_recent(self, weeks_back: int = 1) -> pd.DataFrame:
        cutoff = date.today() - timedelta(weeks=weeks_back)
        query = text("""
            SELECT features
            FROM features.feature_vectors
            WHERE computed_at >= :cutoff
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"cutoff": cutoff}).fetchall()

        if not rows:
            logger.warning("No recent feature vectors found after %s", cutoff)
            return pd.DataFrame()

        import json
        records = [
            r.features if isinstance(r.features, dict) else json.loads(r.features)
            for r in rows
        ]
        return pd.DataFrame(records)

    def compute_and_persist(self, week_start: Optional[date] = None) -> pd.DataFrame:
        """Compute PSI for all numeric features and write to ml.drift_metrics.

        Returns:
            DataFrame of (feature_name, psi_score, status) for this week.
        """
        week_start = week_start or (date.today() - timedelta(days=date.today().weekday()))

        baseline = self._load_baseline()
        recent = self._load_recent()

        if recent.empty:
            logger.warning("No recent data — skipping drift computation")
            return pd.DataFrame()

        numeric_cols = baseline.select_dtypes(include="number").columns.tolist()
        common_cols = [c for c in numeric_cols if c in recent.columns]

        results = []
        for col in common_cols:
            base_vals = baseline[col].dropna().values
            curr_vals = recent[col].dropna().values
            if len(base_vals) < 10 or len(curr_vals) < 10:
                continue
            psi = _psi_score(base_vals, curr_vals)
            status = _psi_status(psi)
            results.append(
                {
                    "feature_name": col,
                    "model_version": self.model_version,
                    "psi_score": round(psi, 6),
                    "status": status,
                    "week_start": week_start,
                }
            )
            if status != "stable":
                logger.warning("DRIFT %s: %s | feature=%s psi=%.4f", status.upper(), col, col, psi)

        if not results:
            return pd.DataFrame()

        df_results = pd.DataFrame(results)
        self._persist_results(df_results)
        return df_results

    def _persist_results(self, df: pd.DataFrame) -> None:
        insert_sql = text("""
            INSERT INTO ml.drift_metrics
                (feature_name, model_version, psi_score, status, week_start)
            VALUES
                (:feature_name, :model_version, :psi_score, :status, :week_start)
            ON CONFLICT (feature_name, model_version, week_start)
            DO UPDATE SET psi_score = EXCLUDED.psi_score, status = EXCLUDED.status
        """)
        with self.engine.begin() as conn:
            conn.execute(insert_sql, df.to_dict("records"))
        logger.info("Persisted %d drift scores to ml.drift_metrics", len(df))
