"""Locust load test for the FinShield scoring API.

Run with:
    locust -f scripts/load_test.py --host http://localhost:8000 --users 50 --spawn-rate 5

Measures p50/p95/p99 latency at 50 and 500 concurrent users.
Target: p95 < 100ms at 500 concurrent users.
"""
from __future__ import annotations

import random

from locust import HttpUser, between, task

SAMPLE_TRANSACTIONS = [
    {
        "transaction_id": f"LOAD_TEST_{i:06d}",
        "customer_id": f"CUST_{i % 1000:04d}",
        "card_id": f"CARD_{i % 500:04d}",
        "merchant_id": f"MERCH_{i % 100:03d}",
        "merchant_category": random.choice(["grocery", "online_retail", "wire_transfer"]),
        "amount": round(random.uniform(10.0, 5000.0), 2),
        "currency": "USD",
        "country": random.choice(["US", "GB", "DE"]),
        "hour_of_day": random.randint(0, 23),
        "day_of_week": random.randint(0, 6),
        "is_weekend": random.choice([True, False]),
    }
    for i in range(1000)
]


class FraudScoringUser(HttpUser):
    """Simulates an analyst or upstream system calling the scoring API."""

    wait_time = between(0.05, 0.5)  # 50–500ms think time

    @task(10)
    def score_transaction(self):
        payload = random.choice(SAMPLE_TRANSACTIONS).copy()
        payload["transaction_id"] = f"LT_{random.randint(0, 999999):06d}"
        self.client.post("/score", json=payload, name="/score")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")

    @task(1)
    def get_metrics(self):
        self.client.get("/metrics/json", name="/metrics/json")
