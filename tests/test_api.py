"""Tests for the FastAPI scoring endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Create a test client with a mock model loaded."""
    from api.main import app

    class MockModel:
        def predict_proba(self, X):
            import numpy as np
            return np.full(len(X), 0.05)

        def explain(self, X, top_n=5):
            return [[{"feature": "amount", "shap_value": 0.1}] for _ in range(len(X))]

    app.state.model = MockModel()
    app.state.model_version = "test_v1"
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_model_loaded(self, client):
        resp = client.get("/health")
        assert resp.json()["model_loaded"] is True


class TestScoreEndpoint:
    VALID_PAYLOAD = {
        "transaction_id": "TEST_001",
        "customer_id": "CUST_001",
        "card_id": "CARD_001",
        "merchant_id": "MERCH_001",
        "amount": 99.99,
        "hour_of_day": 14,
        "day_of_week": 2,
    }

    def test_score_returns_200(self, client):
        resp = client.post("/score", json=self.VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_score_response_fields(self, client):
        resp = client.post("/score", json=self.VALID_PAYLOAD)
        body = resp.json()
        assert "fraud_probability" in body
        assert "risk_tier" in body
        assert "decision" in body
        assert "latency_ms" in body

    def test_score_probability_in_range(self, client):
        resp = client.post("/score", json=self.VALID_PAYLOAD)
        prob = resp.json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_low_fraud_prob_auto_approves(self, client):
        resp = client.post("/score", json=self.VALID_PAYLOAD)
        assert resp.json()["decision"] == "auto_approve"

    def test_invalid_amount_rejected(self, client):
        bad = {**self.VALID_PAYLOAD, "amount": -100.0}
        resp = client.post("/score", json=bad)
        assert resp.status_code == 422

    def test_missing_required_field_rejected(self, client):
        bad = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "amount"}
        resp = client.post("/score", json=bad)
        assert resp.status_code == 422
