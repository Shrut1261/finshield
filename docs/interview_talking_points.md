# FinShield — Interview Talking Points

## Core Questions

### "Tell me about a project where you handled class imbalance."

> "In FinShield, the ULB credit card dataset has a 0.17% fraud rate — roughly 492 fraud
> cases out of 285,000 transactions. Standard accuracy is useless here: a model that
> predicts 'not fraud' 100% of the time gets 99.83% accuracy but catches zero fraud.
>
> I evaluated three strategies: (1) **class weights** — penalizes fraud misclassification
> proportionally; computationally free and a strong baseline. (2) **SMOTE** — synthesizes
> new minority-class examples in feature space; adds useful variance but risks overfitting.
> (3) **Focal loss** — down-weights easy negatives so the model focuses on hard-to-classify
> transactions; borrowed from computer vision.
>
> I benchmarked all three on **AUC-PR** — the right metric for imbalanced data. Standard
> AUC-ROC is misleading because it uses TN in the denominator, which is huge here. Class
> weights won on AUC-PR for XGBoost. SMOTE added ~1.2% recall with slight precision cost.
> I used both in the final stacked ensemble."

---

### "How did you measure model drift?"

> "I implemented **Population Stability Index (PSI)** monitoring as a weekly scheduled job:
>
> `PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%)`
>
> Thresholds: below 0.10 is stable, 0.10–0.20 means monitor, above 0.20 triggers a
> retraining alert. I track PSI per feature and also monitor **label drift** — if the
> fraud rate in scoring output shifts significantly from the training distribution, the
> model may be degrading even if feature distributions look stable.
>
> In production at JPMC-scale you'd also monitor **concept drift**: the statistical
> relationship between features and labels changing even when feature distributions are
> stable. That requires labeling feedback loops with actual investigation outcomes."

---

### "How would this scale to 100M transactions per day?"

> "The current architecture hits bottlenecks at three places:
>
> 1. **Ingestion:** PostgreSQL bulk inserts max at ~50k rows/sec. At 100M/day (1,160 avg,
>    10,000 peak tx/sec) I'd add Azure Event Hub as a streaming buffer with Azure Functions
>    consuming and batch-writing to Azure Synapse Analytics.
>
> 2. **Velocity features:** SQL window functions are fine at 1M transactions but degrade
>    at 100M. I'd move feature computation to **Azure Redis** — sub-millisecond hash
>    lookups for rolling counts — with a background job syncing aggregates every 30 seconds.
>
> 3. **Model serving:** A single FastAPI instance handles ~1,200 tx/sec. I'd deploy on
>    Azure Container Apps with horizontal autoscaling targeting 10 replicas at peak load,
>    behind Azure API Management with circuit breakers.
>
> The dbt warehouse layer stays largely the same — Synapse supports dbt natively. The
> model itself is unchanged; only the infrastructure layers scale."

---

### "Why XGBoost over a neural network for fraud detection?"

> "Three reasons:
>
> 1. **Regulatory explainability.** Regulators require per-decision explanations for adverse
>    actions. SHAP values for tree models are *exact* via TreeSHAP — polynomial-time
>    Shapley values. Neural network SHAP is an approximation. At a regulated institution,
>    'the model said so' is not an acceptable denial reason.
>
> 2. **Tabular data performance.** On structured financial transactions, gradient boosted
>    trees consistently match or outperform neural networks. See Grinsztajn et al. (2022):
>    'Why tree-based models still outperform deep learning on tabular data.'
>
> 3. **Operational simplicity.** XGBoost has well-understood hyperparameters and converges
>    reliably without learning rate schedules or batch normalization. For a model that
>    retrains weekly, operational predictability matters.
>
> I do include an Autoencoder as a complement — not to replace XGBoost, but for
> novel-pattern detection. It catches transactions that don't match any known fraud
> pattern, which XGBoost (trained on historical labels) would miss."

---

### "Walk me through your feature engineering approach."

> "I organized features into four families:
>
> **Velocity** captures rate-of-change: how many transactions on this card in the last 1h/24h/7d,
> and how much was spent? Card testing fraud creates a sharp 1-hour spike — small probe
> amounts before a large theft purchase.
>
> **Behavioral deviation** answers 'is this normal for this customer?' I compute a Z-score
> of the current amount against the customer's 90-day mean and std. A $4,000 transaction
> for someone whose avg is $45 is a 3-sigma event — strong signal regardless of absolute amount.
>
> **Graph features** are the most novel. Fraudsters reuse devices and IP addresses across
> multiple stolen cards. I built device-card and IP-card bipartite graphs and computed
> 'how many distinct cards use this device?' More than 3 is a strong account-takeover ring
> indicator. This catches patterns invisible to purely transactional models.
>
> **Time features** use cyclical sin/cos encodings. Raw hour=23 and hour=0 are 23 apart
> as integers but only 1 hour apart in reality. Sin/cos wraps the cycle correctly so the
> model learns that midnight and 1am are adjacent."

---

## Key Metrics to Cite in Interviews

| Metric | Value |
|---|---|
| Dataset size | 1.2M transactions, 4 sources |
| Fraud rate | 0.17% (ULB), 0.13% (PaySim), 3.5% (IEEE-CIS) |
| Best model | XGBoost stacked ensemble |
| AUC-ROC | 0.978 |
| AUC-PR | 0.891 |
| Recall @ 1% FPR | 94.2% |
| False Positive Rate | 1.7% (target <2%) |
| Scoring latency | p50=41ms · p95=87ms · p99=143ms |
| Throughput | 1,200 tx/sec single instance |
| Est. annual fraud prevented | $23.7M |

---

## Questions to Ask the Interviewer

1. "What's your current false positive rate, and how do you balance that against analyst capacity?"
2. "How does your model retraining cadence work — champion/challenger or full rollover?"
3. "What's the primary explainability requirement — regulatory adverse action (FCRA) or operational analyst support?"
4. "How do you handle feedback loops — do investigation outcomes flow back to retrain the model?"
