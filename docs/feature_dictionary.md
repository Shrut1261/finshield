# FinShield Feature Dictionary

All features are computed by `features/pipeline.py` and persisted to `features.feature_vectors`.

## Velocity Features

| Feature | Type | Description | Window |
|---|---|---|---|
| `tx_count_customer_id_1h` | int | Transactions by customer in last 1 hour | 1h |
| `tx_count_customer_id_24h` | int | Transactions by customer in last 24 hours | 24h |
| `tx_count_customer_id_7d` | int | Transactions by customer in last 7 days | 7d |
| `tx_sum_customer_id_1h` | float | Total $ spent by customer in last 1 hour | 1h |
| `tx_sum_customer_id_24h` | float | Total $ spent by customer in last 24 hours | 24h |
| `tx_sum_customer_id_7d` | float | Total $ spent by customer in last 7 days | 7d |
| `tx_mean_customer_id_1h` | float | Avg $ per transaction, customer, last 1h | 1h |
| `tx_count_card_id_1h` | int | Transactions on card in last 1 hour | 1h |
| `tx_count_card_id_24h` | int | Transactions on card in last 24 hours | 24h |
| `tx_count_card_id_7d` | int | Transactions on card in last 7 days | 7d |
| `tx_sum_card_id_1h` | float | Total $ on card in last 1 hour | 1h |
| `unique_merchants_24h` | int | Distinct merchants used in last 24 hours | 24h |
| `unique_countries_24h` | int | Distinct countries transacted in, last 24h | 24h |
| `unique_merchants_7d` | int | Distinct merchants used in last 7 days | 7d |
| `unique_countries_7d` | int | Distinct countries transacted in, last 7d | 7d |

## Behavioral Features

| Feature | Type | Description |
|---|---|---|
| `amount_zscore` | float | Z-score of current amount vs. customer 90-day mean |
| `amount_ratio_30d` | float | Current amount / customer 30-day average |
| `is_new_merchant_category` | bool | First time customer uses this merchant category |
| `is_new_country` | bool | First time customer transacts in this country |
| `days_since_last_tx` | float | Days elapsed since customer's previous transaction |
| `time_since_last_tx_hours` | float | Hours since last transaction (same card) |
| `is_high_amount` | int | 1 if amount >= 99th percentile of dataset |

## Time Features

| Feature | Type | Description |
|---|---|---|
| `hour_of_day` | int | 0–23 |
| `day_of_week` | int | 0 (Mon) – 6 (Sun) |
| `month` | int | 1–12 |
| `is_weekend` | int | 1 if Saturday or Sunday |
| `is_business_hours` | int | 1 if 9am–5pm, Mon–Fri |
| `hour_sin` | float | Cyclical encoding: sin(2π × hour/24) |
| `hour_cos` | float | Cyclical encoding: cos(2π × hour/24) |
| `dow_sin` | float | Cyclical encoding: sin(2π × dow/7) |
| `dow_cos` | float | Cyclical encoding: cos(2π × dow/7) |
| `month_sin` | float | Cyclical encoding: sin(2π × month/12) |
| `month_cos` | float | Cyclical encoding: cos(2π × month/12) |
| `is_us_holiday` | int | 1 on US federal holidays |

## Graph Features

| Feature | Type | Description |
|---|---|---|
| `device_id_card_count` | int | Distinct cards sharing this device ID |
| `device_id_customer_count` | int | Distinct customers sharing this device |
| `customer_shared_device_flag` | int | 1 if device used by more than 1 customer |
| `ip_address_card_count` | int | Distinct cards sharing this IP address |
| `merchant_fraud_rate` | float | Merchant's rolling fraud rate in this dataset |
| `card_age_days` | float | Days since card's first observed transaction |

## Raw Features

| Feature | Type | Description |
|---|---|---|
| `amount` | float | Transaction amount in USD |

## Feature Store Schema

```sql
CREATE TABLE features.feature_vectors (
    feature_vector_id BIGSERIAL PRIMARY KEY,
    transaction_id    VARCHAR(100) NOT NULL UNIQUE,
    feature_version   VARCHAR(20)  NOT NULL DEFAULT 'v1',
    computed_at       TIMESTAMP    NOT NULL DEFAULT NOW(),
    features          JSONB        NOT NULL
);
```

Versioning allows recomputing features with new logic (`v2`, `v3`) without
breaking production scoring — the API retrieves by `(transaction_id, version)`.
