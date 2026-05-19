"""Tests for ingestion layer."""
from __future__ import annotations

import pandas as pd
import pytest

from ingestion.schema import RiskTier, ScoreResponse, TransactionInput
from ingestion.stream_generator import StreamGenerator


class TestTransactionInput:
    def test_amount_rounded(self):
        tx = TransactionInput(
            transaction_id="T1", customer_id="C1", card_id="K1",
            merchant_id="M1", amount=99.999, hour_of_day=10, day_of_week=2,
        )
        assert tx.amount == 100.0

    def test_currency_uppercased(self):
        tx = TransactionInput(
            transaction_id="T1", customer_id="C1", card_id="K1",
            merchant_id="M1", amount=10.0, hour_of_day=10, day_of_week=2,
            currency="usd",
        )
        assert tx.currency == "USD"

    def test_amount_must_be_positive(self):
        with pytest.raises(Exception):
            TransactionInput(
                transaction_id="T1", customer_id="C1", card_id="K1",
                merchant_id="M1", amount=-5.0, hour_of_day=10, day_of_week=2,
            )

    def test_amount_must_not_exceed_million(self):
        with pytest.raises(Exception):
            TransactionInput(
                transaction_id="T1", customer_id="C1", card_id="K1",
                merchant_id="M1", amount=2_000_000.0, hour_of_day=10, day_of_week=2,
            )


class TestScoreResponse:
    def test_low_prob_auto_approve(self):
        r = ScoreResponse(transaction_id="T1", fraud_probability=0.10, model_version="v1", latency_ms=50.0)
        assert r.risk_tier == RiskTier.LOW
        assert r.decision.value == "auto_approve"

    def test_critical_prob_auto_decline(self):
        r = ScoreResponse(transaction_id="T1", fraud_probability=0.90, model_version="v1", latency_ms=50.0)
        assert r.risk_tier == RiskTier.CRITICAL
        assert r.decision.value == "auto_decline"

    def test_medium_prob_manual_review(self):
        r = ScoreResponse(transaction_id="T1", fraud_probability=0.45, model_version="v1", latency_ms=50.0)
        assert r.risk_tier == RiskTier.MEDIUM
        assert r.decision.value == "manual_review"


class TestStreamGenerator:
    def test_batch_returns_correct_count(self):
        gen = StreamGenerator(fraud_rate=0.01)
        batch = gen.batch(100)
        assert len(batch) == 100

    def test_batch_as_dataframe(self):
        gen = StreamGenerator(fraud_rate=0.01)
        df = gen.batch_as_dataframe(50)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert "transaction_id" in df.columns
        assert "is_fraud" in df.columns

    def test_fraud_rate_approximately_correct(self):
        gen = StreamGenerator(fraud_rate=0.10)
        batch = gen.batch(2000)
        fraud_rate = sum(t.is_fraud for t in batch) / len(batch)
        assert 0.05 <= fraud_rate <= 0.20

    def test_stream_respects_max_transactions(self):
        gen = StreamGenerator(transactions_per_second=10_000.0)
        txs = list(gen.stream(max_transactions=10))
        assert len(txs) == 10
