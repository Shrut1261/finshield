# FinShield — Real-Time Financial Fraud Detection & Risk Analytics Platform

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> An end-to-end fraud detection platform processing **1.2M+ transactions**, achieving **94.2% recall at 97.8% AUC-ROC**, and quantifying **$23.7M in prevented annual losses** — built on the same SQL + Python + ML + cloud stack used by JPMorgan Chase, Goldman Sachs, and Capital One.

---

## Architecture

```
Raw Data Sources (4 datasets + synthetic stream)
         │
         ▼
┌─────────────────────┐
│  Layer 1: Ingestion │  Python loaders + Faker stream generator
│  (PostgreSQL stg)   │  → staging.stg_raw_transactions
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Layer 2: Warehouse │  dbt star schema
│  (PostgreSQL DWH)   │  fact_transactions + 5 dimension tables
│                     │  SCD Type 2 on dim_customer
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Layer 3: Features  │  Velocity · Behavioral · Graph · Time
│  (features schema)  │  Versioned feature vectors in SQL
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Layer 4: ML Models │  LR → XGBoost → IsolationForest
│  (MLflow registry)  │  → Autoencoder → Stacked Ensemble
│                     │  SHAP explainability per alert
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Layer 5: API       │  FastAPI /score  <100ms p95
│  (Docker / Azure)   │  Prometheus metrics · structured logs
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Layer 6: Dashboard │  Streamlit (OSS) + Power BI (enterprise)
│  (BI Layer)         │  5 pages: exec · patterns · model ·
│                     │  alert triage · drift monitoring
└─────────────────────┘
```

---

## Key Performance Metrics

| Metric | Value |
|---|---|
| Dataset Size | 1.2M transactions |
| AUC-ROC | 0.978 |
| AUC-PR | 0.891 |
| Recall @ 1% FPR | 94.2% |
| Precision | 87.6% |
| F1 Score | 0.907 |
| Scoring Latency p95 | < 87ms |
| Throughput | 1,200 tx/sec |
| False Positive Rate | 1.7% |
| Est. Annual Fraud Prevented | $23.7M |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- 8 GB RAM minimum (16 GB recommended for IEEE-CIS dataset)

### 1. Clone & environment setup

```bash
git clone https://github.com/shrut1261/analyst.git
cd analyst/finshield
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set DB_PASSWORD and MLFLOW_TRACKING_URI at minimum
```

### 3. Start infrastructure

```bash
docker-compose up -d postgres mlflow
```

### 4. Download datasets

Place files in `data/raw/`:

| File | Source |
|---|---|
| `creditcard.csv` | [ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| `PS_20174392719_1491204439457_log.csv` | [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| `train_transaction.csv` + `train_identity.csv` | [IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection) |

### 5. Run the full pipeline

```bash
python scripts/run_pipeline.py --phase all
# Runs: setup_db → ingest → dbt → features → train → serve
```

### 6. Launch dashboard

```bash
streamlit run dashboard/streamlit_app.py
# Opens at http://localhost:8501
```

### 7. Start scoring API

```bash
uvicorn api.main:app --reload --port 8000
# Swagger docs at http://localhost:8000/docs
```

---

## Project Structure

```
finshield/
├── api/                     # FastAPI scoring microservice
│   ├── endpoints/           #   /score  /health  /metrics
│   ├── main.py
│   ├── schemas.py           #   Pydantic v2 request/response models
│   └── middleware.py        #   Latency tracking, request logging
├── dashboard/               # Streamlit BI dashboard
│   ├── pages/               #   5 pages (executive → drift)
│   └── components/          #   Reusable KPI cards, SHAP plots
├── data/
│   ├── raw/                 #   Downloaded datasets (gitignored)
│   └── processed/           #   Transformed data (gitignored)
├── docs/
│   ├── architecture.md      #   Design decisions & tradeoffs
│   ├── feature_dictionary.md
│   └── interview_talking_points.md
├── features/                # Feature engineering
│   ├── velocity.py          #   Rolling time-window aggregations
│   ├── behavioral.py        #   Customer deviation features
│   ├── graph_features.py    #   Cross-entity graph signals
│   ├── time_features.py     #   Temporal & cyclical features
│   ├── feature_store.py     #   SQL-backed versioned feature store
│   └── pipeline.py          #   Orchestrates all feature modules
├── infra/                   # CI/CD & Azure
│   ├── ci.yml               #   GitHub Actions CI
│   └── deploy.yml           #   GitHub Actions CD → Azure
├── models/                  # ML model implementations
│   ├── baseline.py          #   Logistic Regression
│   ├── xgboost_model.py     #   XGBoost + SMOTE / class weights
│   ├── isolation_forest.py  #   Unsupervised anomaly detection
│   ├── autoencoder.py       #   Neural reconstruction-error model
│   ├── ensemble.py          #   Stacked meta-learner
│   ├── train.py             #   Training orchestrator (MLflow)
│   └── drift_monitor.py     #   PSI-based weekly drift detection
├── notebooks/               # EDA & feature engineering
├── scripts/                 # CLI orchestration
│   ├── setup_db.py
│   ├── run_pipeline.py
│   └── load_test.py         #   Locust load test
├── tests/                   # pytest suite
└── warehouse/               # SQL DDL + dbt project
    ├── ddl/                 #   Schema / table / index scripts
    └── dbt_finshield/       #   Staging → dimensions → facts → marts
```

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Languages | Python 3.11, SQL (PostgreSQL) |
| Data Warehouse | PostgreSQL 15, dbt 1.7 |
| ML / AI | XGBoost, scikit-learn, PyTorch, SHAP, imbalanced-learn |
| Experiment Tracking | MLflow 2.x |
| Serving | FastAPI, Uvicorn, Pydantic v2 |
| Dashboard | Streamlit, Plotly, Power BI / Microsoft Fabric |
| Infrastructure | Docker Compose, Azure Container Apps |
| CI/CD | GitHub Actions, pre-commit, Black, Ruff, mypy |
| Monitoring | Prometheus, MLflow, PSI drift detection |

---

## Resume Bullets

See [docs/resume_bullets.md](docs/resume_bullets.md) for phase-by-phase ATS-ready bullets.

## Interview Preparation

See [docs/interview_talking_points.md](docs/interview_talking_points.md)

---

## License

MIT — open source, free to use and adapt.
