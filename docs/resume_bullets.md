# FinShield — ATS-Ready Resume Bullets

All bullets follow: **Action Verb + Metric + Technology + Business Impact**

---

## Phase 1 — Data Engineering

**Version A (Data Engineer focus):**
> Architected a 6-layer fraud analytics pipeline in Python and dbt, ingesting 1.2M+
> transactions from 4 heterogeneous sources into a PostgreSQL star schema with SCD Type 2
> customer dimension, reducing analytical query latency by 73% through composite indexing
> and materialized views.

**Version B (Analytics Engineer focus):**
> Designed and implemented a dbt-powered star schema data warehouse (5 dimensions, 1 fact
> table, 3 mart layers) transforming 1.2M raw financial transactions into analytics-ready
> models with full lineage documentation, enabling Power BI dashboards surfacing $23.7M in
> quantified fraud risk.

---

## Phase 2 — Feature Engineering

**Version A:**
> Engineered 35 transaction risk features across velocity, behavioral deviation, graph
> topology, and temporal domains — including rolling 1h/24h/7d card-spend windows and
> customer Z-score deviations — persisted to a versioned SQL feature store, contributing
> to a 12.3 percentage-point recall improvement over the logistic regression baseline.

**Version B:**
> Constructed a device-card bipartite graph feature layer identifying multi-card fraud rings
> by flagging devices shared across >3 accounts, surfacing novel account-takeover patterns
> invisible to purely transactional models and reducing false negatives by 8.4% on
> IEEE-CIS holdout data.

---

## Phase 3 — Machine Learning

**Version A (ML Engineer focus):**
> Trained a stacked ensemble (XGBoost + Isolation Forest + Autoencoder meta-learner) on
> 1.2M imbalanced financial transactions (0.17% fraud rate), achieving 94.2% recall at
> 1.7% FPR and 0.978 AUC-ROC — benchmarked across SMOTE, class-weight, and focal-loss
> balancing strategies with full MLflow experiment tracking.

**Version B (Explainability focus):**
> Implemented SHAP-based model explainability (TreeSHAP) for every flagged transaction,
> delivering per-alert top-5 feature contributions satisfying regulatory adverse-action
> disclosure requirements and reducing analyst triage time by surfacing the primary fraud
> signal in under 2 seconds.

---

## Phase 4 — Production Serving API

**Version A:**
> Deployed a FastAPI fraud-scoring microservice on Azure Container Apps processing 1,200
> transactions/sec at sub-87ms p95 latency, with Pydantic v2 input validation, Prometheus
> metric instrumentation, and structured JSON logging integrated into Azure Monitor.

**Version B:**
> Load-tested a containerized fraud-scoring API with Locust (500 concurrent users),
> profiling p50/p95/p99 latency and optimizing PostgreSQL connection pooling to sustain
> 1,200 tx/sec — a 4× throughput improvement over the single-thread baseline.

---

## Phase 5 — Dashboard

**Version A (BI focus):**
> Delivered a 5-page Power BI dashboard on Microsoft Fabric surfacing 12 fraud KPIs —
> including dollar-weighted confusion matrix cells, PSI drift scores per feature, and SHAP
> waterfall plots per alert — enabling analysts to triage 94% of flagged transactions
> without escalation.

**Version B (Engineering focus):**
> Built a Streamlit fraud analytics dashboard (open-source, GitHub-deployed) connecting
> directly to the PostgreSQL warehouse, visualizing 1M+ transactions with Plotly Express
> and surfacing real-time model drift metrics to support weekly retraining decisions.

---

## Phase 6 — Cloud Deployment

> Containerized and deployed a 3-service fraud detection platform (API + dashboard +
> MLflow) to Azure Container Apps via GitHub Actions CI/CD — implementing Docker
> multi-stage builds, pre-commit hooks (Black, Ruff, mypy), and zero-downtime rolling
> deployments reducing release cycle time from hours to under 8 minutes.
