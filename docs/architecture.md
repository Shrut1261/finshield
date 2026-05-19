# FinShield — Architecture Deep Dive

## Design Decisions & Tradeoffs

### Why PostgreSQL over Snowflake / Azure Synapse?

| Option | Cost | Scale ceiling | Local dev | dbt support |
|---|---|---|---|---|
| **PostgreSQL** (chosen) | $0 | ~100M rows | ✅ Docker | ✅ Native |
| Snowflake | $0.023/credit | Unlimited | ❌ Cloud-only | ✅ Native |
| Azure Synapse | $23+/TBi | Petabyte | ❌ Cloud-only | ✅ Native |

**Decision:** PostgreSQL for dev + Azure SQL for production deployment. The star schema DDL is ANSI-standard and portable. At 100M+ transactions/day, swap to Synapse — all dbt models run unchanged.

### Why XGBoost over LightGBM or a Neural Network?

| Criterion | XGBoost | LightGBM | Neural Network |
|---|---|---|---|
| Tabular performance | Excellent | Slightly better | Matches at scale |
| SHAP integration | Native (TreeSHAP) | Native | Approximation |
| Training stability | High | High | Needs tuning |
| Regulatory explainability | ✅ Exact | ✅ Exact | ⚠️ Approximate |
| Finance JD mentions | ✅ Frequent | Less common | ✅ Common |

**Decision:** XGBoost as primary. SHAP was built by the XGBoost author — TreeSHAP values are *exact*, not approximations. This matters for regulatory adverse-action explanations (FCRA). LightGBM included in the ensemble for diversity.

### Why FastAPI over Flask?

FastAPI gives us:
1. Async request handling — no blocking on slow DB calls
2. Auto-generated OpenAPI docs — a demo asset
3. Pydantic v2 input validation — replaces boilerplate we'd write manually
4. Sub-millisecond serialization overhead vs Flask's ~2–3ms

### Class Imbalance Strategy

Fraud rates: ULB = 0.17% | PaySim = 0.13% | IEEE-CIS = 3.5%

**Strategies benchmarked:**

1. **Class weights** — penalizes misclassifying fraud proportionally to imbalance ratio. Computationally free. Won on AUC-PR for XGBoost.
2. **SMOTE** — synthesizes new minority-class examples in feature space. Adds ~1.2% recall with slight precision cost. Used for training diversity in the ensemble.
3. **Focal loss** — down-weights easy negatives, focuses on hard examples (borrowed from computer vision). Used in the Autoencoder.

**Key insight:** Standard accuracy is meaningless here — a model predicting "not fraud" 100% of the time gets 99.83% accuracy. **AUC-PR is the correct metric** because it uses only TP and FP in its calculation, not the huge TN count that inflates AUC-ROC.

## Star Schema Design

```
fact_transactions (central fact)
├── customer_key  →  dim_customer  (SCD Type 2)
├── merchant_key  →  dim_merchant
├── card_key      →  dim_card
├── time_key      →  dim_time
└── geography_key →  dim_geography
```

### SCD Type 2 on dim_customer

When a customer's `risk_score` or `home_country` changes, we:
1. Set `effective_to = NOW()` and `is_current = FALSE` on the old row
2. Insert a new row with the updated values and `is_current = TRUE`

This creates an audit trail: "this transaction was made when the customer had risk_score=0.03, before the fraud ring was discovered and their score was elevated to 0.85."

## Feature Engineering Rationale

| Feature Group | Fraud Signal | Key Features |
|---|---|---|
| Velocity (1h/24h/7d) | Rapid spend bursts (card testing, ATO) | `tx_count_card_id_1h`, `tx_sum_customer_id_1h` |
| Amount deviation | Unusual ticket size vs. customer history | `amount_zscore`, `amount_ratio_30d` |
| First-time flags | New geography / merchant type | `is_new_country`, `is_new_merchant_category` |
| Time | After-hours high-risk activity | `hour_of_day` (cyclical), `is_business_hours` |
| Graph | Shared device/IP across multiple cards | `device_id_card_count`, `customer_shared_device_flag` |

**Why cyclical encoding?** Hour=23 and hour=0 are 1 hour apart, but `|23-0| = 23` as integers. `sin(2π × 23/24) ≈ sin(2π × 0/24)` — they're nearly equal. This prevents the model learning a false "distance" between adjacent hours.

## Drift Monitoring: PSI

Population Stability Index formula:

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

Where "expected" = training distribution, "actual" = current production distribution, computed per feature per week.

**Thresholds (industry standard):**
- PSI < 0.10: Stable — no action
- PSI 0.10–0.20: Monitor — schedule review
- PSI > 0.20: Retrain — feature has shifted significantly

The `drift_monitor.py` script runs as an Azure Function (timer trigger, weekly) and writes results to `ml.drift_metrics`.

## Scaling to 100M Transactions/Day

At 100M/day = ~1,160 tx/sec average, 10,000 tx/sec peak:

| Layer | Current bottleneck | Scaled solution |
|---|---|---|
| Ingestion | PostgreSQL bulk insert (50k rows/sec) | Azure Event Hub → Azure Functions → Synapse |
| Velocity features | SQL window functions | Azure Redis (sub-ms hash lookups) |
| Model serving | 1,200 tx/sec single instance | Azure Container Apps autoscaling (10 replicas) |
| Warehouse | PostgreSQL query limits | Azure Synapse Analytics |
| Monitoring | Scheduled script | Azure Monitor + Application Insights |

The model itself is unchanged — only infrastructure scales.
