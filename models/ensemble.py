"""Stacked ensemble: combines XGBoost, IsolationForest, and Autoencoder.

Meta-learner is a simple Logistic Regression trained on out-of-fold
predictions from the base models.  This avoids overfitting the meta-learner
to the same data the base models saw.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from models.autoencoder import AutoencoderDetector
from models.baseline import BaselineModel
from models.isolation_forest import IsolationForestDetector
from models.xgboost_model import XGBoostFraudModel

logger = logging.getLogger(__name__)


class FraudEnsemble:
    """Stacked ensemble that combines predictions from all base models.

    Architecture:
      Layer 1 (base models): XGBoost, IsolationForest, Autoencoder, LR baseline
      Layer 2 (meta-learner): LogisticRegression on OOF base-model probabilities

    The ensemble score is a weighted blend: 0.5 × XGBoost + 0.3 × meta-learner
    + 0.2 × IsolationForest.  Weights were calibrated on validation AUC-PR.
    """

    XGB_WEIGHT = 0.50
    IF_WEIGHT = 0.20
    META_WEIGHT = 0.30

    def __init__(self, n_folds: int = 5, random_state: int = 42) -> None:
        self.n_folds = n_folds
        self.random_state = random_state
        self.xgb = XGBoostFraudModel(strategy="class_weight")
        self.iso = IsolationForestDetector()
        self.ae = AutoencoderDetector(epochs=20)
        self.baseline = BaselineModel()
        self.meta: Optional[LogisticRegression] = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FraudEnsemble":
        """Train all base models and fit the meta-learner on OOF predictions."""
        logger.info("Training ensemble on %d samples", len(X))

        # ── Train base models on full training set ─────────────────────────────
        self.xgb.fit(X, y)
        self.iso.fit(X)
        self.ae.fit(X, y)
        self.baseline.fit(X, y)

        # ── OOF meta-features ─────────────────────────────────────────────────
        oof_xgb = np.zeros(len(y))
        oof_iso = np.zeros(len(y))
        oof_ae = np.zeros(len(y))
        oof_lr = np.zeros(len(y))

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y.iloc[train_idx]

            fold_xgb = XGBoostFraudModel(strategy="class_weight")
            fold_xgb.fit(X_tr, y_tr)
            oof_xgb[val_idx] = fold_xgb.predict_proba(X_val)

            fold_iso = IsolationForestDetector()
            fold_iso.fit(X_tr)
            oof_iso[val_idx] = fold_iso.anomaly_score(X_val)

            fold_ae = AutoencoderDetector(epochs=10)
            fold_ae.fit(X_tr, y_tr)
            oof_ae[val_idx] = fold_ae.anomaly_score(X_val)

            fold_lr = BaselineModel()
            fold_lr.fit(X_tr, y_tr)
            oof_lr[val_idx] = fold_lr.predict_proba(X_val)

            logger.info("Fold %d/%d complete", fold + 1, self.n_folds)

        meta_X = np.column_stack([oof_xgb, oof_iso, oof_ae, oof_lr])
        self.meta = LogisticRegression(
            C=1.0, class_weight="balanced", random_state=self.random_state, max_iter=500
        )
        self.meta.fit(meta_X, y)
        self._fitted = True
        logger.info("Ensemble meta-learner fitted")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self.meta is None:
            raise RuntimeError("Ensemble not fitted.")

        p_xgb = self.xgb.predict_proba(X)
        p_iso = self.iso.anomaly_score(X)
        p_ae = self.ae.anomaly_score(X)
        p_lr = self.baseline.predict_proba(X)

        meta_X = np.column_stack([p_xgb, p_iso, p_ae, p_lr])
        p_meta = self.meta.predict_proba(meta_X)[:, 1]

        # Weighted blend
        return np.clip(
            self.XGB_WEIGHT * p_xgb + self.IF_WEIGHT * p_iso + self.META_WEIGHT * p_meta,
            0.0,
            1.0,
        )

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def explain(self, X: pd.DataFrame, top_n: int = 5) -> list[list[dict[str, float]]]:
        """Delegate SHAP explanations to the XGBoost base model."""
        return self.xgb.explain(X, top_n=top_n)
