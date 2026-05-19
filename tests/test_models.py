"""Tests for ML model implementations."""
from __future__ import annotations

import numpy as np
import pytest


class TestBaselineModel:
    def test_fit_and_predict(self, feature_matrix):
        from models.baseline import BaselineModel
        X, y = feature_matrix
        model = BaselineModel()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert len(proba) == len(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_binary(self, feature_matrix):
        from models.baseline import BaselineModel
        X, y = feature_matrix
        model = BaselineModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_unfitted_raises(self, feature_matrix):
        from models.baseline import BaselineModel
        X, _ = feature_matrix
        model = BaselineModel()
        with pytest.raises(RuntimeError):
            model.predict_proba(X)


class TestXGBoostModel:
    def test_fit_class_weight(self, feature_matrix):
        from models.xgboost_model import XGBoostFraudModel
        X, y = feature_matrix
        model = XGBoostFraudModel(strategy="class_weight", n_estimators=10)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert len(proba) == len(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_feature_importance_sums_to_one(self, feature_matrix):
        from models.xgboost_model import XGBoostFraudModel
        X, y = feature_matrix
        model = XGBoostFraudModel(strategy="class_weight", n_estimators=10)
        model.fit(X, y)
        importance = model.get_feature_importance()
        assert abs(importance.sum() - 1.0) < 0.01


class TestIsolationForest:
    def test_anomaly_score_range(self, feature_matrix):
        from models.isolation_forest import IsolationForestDetector
        X, _ = feature_matrix
        model = IsolationForestDetector(n_estimators=10)
        model.fit(X)
        scores = model.anomaly_score(X)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_predict_binary(self, feature_matrix):
        from models.isolation_forest import IsolationForestDetector
        X, _ = feature_matrix
        model = IsolationForestDetector(n_estimators=10)
        model.fit(X)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})
