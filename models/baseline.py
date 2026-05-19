"""Logistic Regression baseline model.

Deliberately simple — establishes the performance floor.
In interviews, explaining *why* baseline exists (benchmarking, sanity check,
regulatory audit trail) matters as much as the model itself.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BaselineModel:
    """Logistic Regression with class-weight balancing and feature scaling."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42) -> None:
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model: Optional[Pipeline] = None

    def build(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=self.C,
                class_weight="balanced",
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver="lbfgs",
                n_jobs=-1,
            )),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineModel":
        logger.info("Training baseline LR on %d samples (fraud_rate=%.4f%%)", len(y), y.mean() * 100)
        self.model = self.build()
        self.model.fit(X, y)
        logger.info("Baseline LR training complete")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
