"""Tests for feature engineering modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestTimeFeatures:
    def test_cyclical_encodings_range(self, sample_transactions):
        from features.time_features import compute_time_features
        df = compute_time_features(sample_transactions, timestamp_col="timestamp")
        assert df["hour_sin"].between(-1.01, 1.01).all()
        assert df["hour_cos"].between(-1.01, 1.01).all()

    def test_is_weekend_correct(self, sample_transactions):
        from features.time_features import compute_time_features
        df = compute_time_features(sample_transactions, timestamp_col="timestamp")
        ts = pd.to_datetime(sample_transactions["timestamp"])
        expected_weekend = (ts.dt.dayofweek >= 5).astype("int8")
        pd.testing.assert_series_equal(
            df["is_weekend"].reset_index(drop=True),
            expected_weekend.reset_index(drop=True),
            check_names=False,
        )

    def test_all_expected_columns_present(self, sample_transactions):
        from features.time_features import compute_time_features
        df = compute_time_features(sample_transactions, timestamp_col="timestamp")
        expected = ["hour_of_day", "day_of_week", "is_weekend", "is_business_hours",
                    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_us_holiday"]
        for col in expected:
            assert col in df.columns, f"Missing: {col}"


class TestGraphFeatures:
    def test_merchant_fraud_rate_non_negative(self, sample_transactions):
        from features.graph_features import compute_graph_features
        df = compute_graph_features(sample_transactions)
        assert (df["merchant_fraud_rate"] >= 0).all()

    def test_device_card_count_at_least_one(self, sample_transactions):
        from features.graph_features import compute_graph_features
        df = compute_graph_features(sample_transactions)
        assert (df["device_id_card_count"] >= 1).all()

    def test_shared_device_flag_binary(self, sample_transactions):
        from features.graph_features import compute_graph_features
        df = compute_graph_features(sample_transactions)
        assert df["customer_shared_device_flag"].isin([0, 1]).all()


class TestBehavioralFeatures:
    def test_zscore_exists(self, sample_transactions):
        from features.behavioral import compute_behavioral_features
        df = compute_behavioral_features(sample_transactions)
        assert "amount_zscore" in df.columns

    def test_is_high_amount_binary(self, sample_transactions):
        from features.behavioral import compute_behavioral_features
        df = compute_behavioral_features(sample_transactions)
        assert df["is_high_amount"].isin([0, 1]).all()

    def test_no_nan_in_core_features(self, sample_transactions):
        from features.behavioral import compute_behavioral_features
        df = compute_behavioral_features(sample_transactions)
        core = ["amount_zscore", "days_since_last_tx", "is_new_merchant_category"]
        for col in core:
            assert df[col].notna().all(), f"NaN in {col}"
