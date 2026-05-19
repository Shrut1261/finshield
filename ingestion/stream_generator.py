"""Synthetic real-time transaction stream generator using Faker.

Produces a configurable mix of legitimate and fraudulent transactions
with realistic behavioral profiles per customer.  Used for:
  - Testing the scoring API under load
  - Simulating streaming ingestion for demo purposes
  - Generating additional training data augmentation
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Generator, Optional

import numpy as np
from faker import Faker

fake = Faker()
Faker.seed(42)
rng = np.random.default_rng(42)

MERCHANT_CATEGORIES = [
    "grocery",
    "gas_station",
    "restaurant",
    "online_retail",
    "travel",
    "entertainment",
    "healthcare",
    "atm_withdrawal",
    "wire_transfer",
    "crypto",
]

HIGH_RISK_CATEGORIES = frozenset({"wire_transfer", "crypto", "atm_withdrawal"})
COUNTRIES = ["US", "GB", "CA", "DE", "FR", "AU", "SG", "JP", "BR", "MX"]


@dataclass
class SyntheticTransaction:
    transaction_id: str
    customer_id: str
    card_id: str
    merchant_id: str
    merchant_category: str
    amount: float
    currency: str
    country: str
    city: str
    latitude: float
    longitude: float
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    device_id: Optional[str]
    ip_address: Optional[str]
    timestamp: datetime
    is_fraud: int = 0
    fraud_scenario: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ── Fraud scenario injectors ──────────────────────────────────────────────────

def _scenario_card_testing(tx: SyntheticTransaction) -> SyntheticTransaction:
    """Tiny probe amount before a large fraudulent purchase."""
    tx.amount = round(float(rng.uniform(0.01, 4.99)), 2)
    tx.merchant_category = "online_retail"
    tx.is_fraud = 1
    tx.fraud_scenario = "card_testing"
    return tx


def _scenario_geo_anomaly(tx: SyntheticTransaction) -> SyntheticTransaction:
    """Transaction from a country far from the customer's home."""
    tx.country = fake.country_code()
    tx.city = fake.city()
    tx.latitude = float(fake.latitude())
    tx.longitude = float(fake.longitude())
    tx.is_fraud = 1
    tx.fraud_scenario = "geo_anomaly"
    return tx


def _scenario_after_hours_large(tx: SyntheticTransaction) -> SyntheticTransaction:
    """Unusually large amount during 1–4am hours."""
    tx.amount = round(float(rng.uniform(500.0, 15_000.0)), 2)
    tx.hour_of_day = int(rng.integers(1, 5))
    tx.merchant_category = random.choice(list(HIGH_RISK_CATEGORIES))
    tx.is_fraud = 1
    tx.fraud_scenario = "after_hours_large"
    return tx


def _scenario_account_takeover(tx: SyntheticTransaction) -> SyntheticTransaction:
    """Rapid spend from a new device the customer has never used."""
    tx.device_id = str(uuid.uuid4())
    tx.ip_address = fake.ipv4_public()
    tx.amount = round(float(rng.uniform(200.0, 3_000.0)), 2)
    tx.is_fraud = 1
    tx.fraud_scenario = "account_takeover"
    return tx


FRAUD_SCENARIOS: list[Callable[[SyntheticTransaction], SyntheticTransaction]] = [
    _scenario_card_testing,
    _scenario_geo_anomaly,
    _scenario_after_hours_large,
    _scenario_account_takeover,
]


# ── Generator class ────────────────────────────────────────────────────────────

class StreamGenerator:
    """Generates a continuous stream of transactions with configurable fraud injection.

    Args:
        fraud_rate: Fraction of transactions to mark as fraudulent (default 0.0017 = 0.17%).
        transactions_per_second: Rate limit for the ``stream()`` generator.
        n_customers: Size of the synthetic customer population.
        n_merchants: Size of the synthetic merchant population.
    """

    def __init__(
        self,
        fraud_rate: float = 0.0017,
        transactions_per_second: float = 100.0,
        n_customers: int = 10_000,
        n_merchants: int = 2_000,
    ) -> None:
        self.fraud_rate = fraud_rate
        self.tps = transactions_per_second
        self.customers = [f"CUST_{i:06d}" for i in range(n_customers)]
        self.merchants = [f"MERCH_{i:05d}" for i in range(n_merchants)]
        self._profiles = self._build_customer_profiles()

    def _build_customer_profiles(self) -> dict[str, dict]:
        """Build stable per-customer behavioral profiles for realistic anomaly injection."""
        profiles: dict[str, dict] = {}
        for cust in self.customers:
            profiles[cust] = {
                "home_country": rng.choice(COUNTRIES),
                "typical_amount_mean": float(rng.uniform(20.0, 500.0)),
                "typical_amount_std": float(rng.uniform(5.0, 100.0)),
                "preferred_categories": rng.choice(MERCHANT_CATEGORIES, size=3).tolist(),
                "active_hours": rng.choice(range(8, 22), size=6, replace=False).tolist(),
            }
        return profiles

    def _make_transaction(self, now: Optional[datetime] = None) -> SyntheticTransaction:
        now = now or datetime.utcnow()
        customer_id = random.choice(self.customers)
        profile = self._profiles[customer_id]

        amount = max(
            0.01,
            float(rng.normal(profile["typical_amount_mean"], profile["typical_amount_std"])),
        )

        tx = SyntheticTransaction(
            transaction_id=str(uuid.uuid4()),
            customer_id=customer_id,
            card_id=f"CARD_{abs(hash(customer_id)) % 50_000:06d}",
            merchant_id=random.choice(self.merchants),
            merchant_category=random.choice(profile["preferred_categories"]),
            amount=round(amount, 2),
            currency="USD",
            country=profile["home_country"],
            city=fake.city(),
            latitude=float(fake.latitude()),
            longitude=float(fake.longitude()),
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            is_weekend=now.weekday() >= 5,
            device_id=f"DEV_{abs(hash(customer_id + now.date().isoformat())) % 100_000:06d}",
            ip_address=fake.ipv4_public(),
            timestamp=now,
        )

        if rng.random() < self.fraud_rate:
            scenario = random.choice(FRAUD_SCENARIOS)
            tx = scenario(tx)

        return tx

    def stream(
        self,
        max_transactions: Optional[int] = None,
    ) -> Generator[SyntheticTransaction, None, None]:
        """Yield transactions at `transactions_per_second` rate.

        Set max_transactions=None for an infinite stream.
        """
        count = 0
        interval = 1.0 / self.tps
        while max_transactions is None or count < max_transactions:
            yield self._make_transaction()
            count += 1
            time.sleep(interval)

    def batch(self, n: int) -> list[SyntheticTransaction]:
        """Generate n transactions without rate limiting (for bulk seeding)."""
        return [self._make_transaction() for _ in range(n)]

    def batch_as_dataframe(self, n: int) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame([tx.to_dict() for tx in self.batch(n)])


if __name__ == "__main__":
    gen = StreamGenerator(fraud_rate=0.01, transactions_per_second=10.0)
    print("Generating 5 sample transactions:")
    for tx in gen.stream(max_transactions=5):
        label = "FRAUD" if tx.is_fraud else "legit"
        print(f"  [{label}] {tx.transaction_id[:8]}... ${tx.amount:.2f} "
              f"@ {tx.merchant_category} | {tx.country}")
