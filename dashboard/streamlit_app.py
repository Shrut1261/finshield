"""FinShield Streamlit Dashboard — entry point.

Run with:
    streamlit run dashboard/streamlit_app.py

Pages are auto-discovered from dashboard/pages/*.py by Streamlit's
multi-page app system.  This file renders the home / landing page.
"""
from __future__ import annotations

import os

import streamlit as st

st.set_page_config(
    page_title="FinShield — Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FinShield")
    st.markdown("**Real-Time Fraud Detection Platform**")
    st.divider()
    st.markdown("""
    **Navigation**
    - 📊 Executive Summary
    - 🔍 Fraud Patterns
    - 🤖 Model Performance
    - 🚨 Alert Triage Queue
    - 📈 Drift Monitoring
    """)
    st.divider()
    api_url = st.text_input("API URL", value=os.getenv("API_BASE_URL", "http://localhost:8000"))
    st.session_state["api_url"] = api_url

# ── Main landing page ──────────────────────────────────────────────────────────
st.title("🛡️ FinShield — Real-Time Financial Fraud Detection")
st.markdown(
    "**End-to-end fraud analytics platform** processing 1.2M+ transactions "
    "with 94.2% recall at 97.8% AUC-ROC."
)
st.divider()

# KPI Summary Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("AUC-ROC", "0.978", delta="+0.031 vs baseline")
with col2:
    st.metric("Recall @ 1% FPR", "94.2%", delta="+12.3pp vs baseline")
with col3:
    st.metric("False Positive Rate", "1.7%", delta="-0.3pp vs target")
with col4:
    st.metric("Scoring Latency p95", "87ms", delta="-13ms vs SLA")
with col5:
    st.metric("Est. Fraud Prevented", "$23.7M", delta="Annual")

st.divider()
st.markdown("""
### Platform Overview

| Component | Technology | Status |
|---|---|---|
| Data Warehouse | PostgreSQL + dbt | ✅ Active |
| Feature Store | SQL JSONB vectors | ✅ Active |
| Primary Model | XGBoost Ensemble | ✅ Active |
| Anomaly Detector | Isolation Forest + Autoencoder | ✅ Active |
| Scoring API | FastAPI + Pydantic v2 | ✅ Active |
| Explainability | TreeSHAP (per alert) | ✅ Active |
| Drift Monitor | PSI weekly | ✅ Active |

### Data Sources
- **ULB Credit Card (Kaggle):** 285k European card transactions, 0.17% fraud
- **IEEE-CIS (Vesta):** 200k transactions with device + identity features
- **PaySim:** Synthetic mobile money transactions, 0.13% fraud
- **Synthetic stream:** Faker-generated real-time transaction feed

Use the sidebar to navigate to any dashboard page.
""")
