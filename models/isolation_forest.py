"""Isolation Forest anomaly detector.

Complements the supervised XGBoost model by flagging transactions that
are simply unusual — even if they don't match any known fraud pattern.
This is the 'unknown unknowns' detector in the ensemble.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """Wraps sklearn IsolationForest with calibrated anomaly scores.

    The raw IsolationForest score is an anomaly score in (-inf, 0], where
    more negative = more anomalous.  We rescale to [0, 1] so it can be
    combined with the XGBoost probability in the ensemble.
    """

    def __init__(
        self,
        contamination: float = 0.002,
        n_estimators: int = 200,
        max_samples: int = 512,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self._score_min: float = -1.0
        self._score_max: float = 0.0

    def fit(self, X: pd.DataFrame) -> "IsolationForestDetector":
        """Fit scaler and isolation forest on the full (unlabeled) dataset."""
        logger.info("Fitting IsolationForest on %d samples", len(X))
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

        # Calibrate score range on training data
        raw_scores = self.model.score_samples(X_scaled)
        self._score_min = float(raw_scores.min())
        self._score_max = float(raw_scores.max())
        logger.info(
            "IsolationForest fitted. Score range: [%.4f, %.4f]",
            self._score_min,
            self._score_max,
        )
        return self

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated anomaly probability in [0, 1].

        Higher score = more anomalous = higher fraud probability.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted.")
        X_scaled = self.scaler.transform(X)
        raw = self.model.score_samples(X_scaled)
        # Invert and normalize: more negative raw score → higher anomaly score
        score_range = self._score_max - self._score_min
        normalized = (self._score_max - raw) / max(score_range, 1e-9)
        return np.clip(normalized, 0.0, 1.0)

    def predict(self, X: pd.DataFrame, threshold: float = 0.7) -> np.ndarray:
        """Binary prediction: 1 = anomaly, 0 = normal."""
        return (self.anomaly_score(X) >= threshold).astype(int)
