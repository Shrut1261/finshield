"""Executive Summary dashboard page.

KPIs: total transactions, fraud rate, $ prevented, detection quality.
Connects to the mart_fraud_kpis dbt mart for pre-aggregated data.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Executive Summary", page_icon="📊", layout="wide")

st.title("📊 Executive Summary")
st.caption(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")


@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    """Load from mart_fraud_kpis or fall back to demo data."""
    db_url = os.getenv("DATABASE_URL", "")
    if db_url:
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(db_url)
            with engine.connect() as conn:
                df = pd.read_sql(
                    "SELECT * FROM warehouse.mart_fraud_kpis ORDER BY day_bucket DESC LIMIT 90",
                    conn,
                )
            return df
        except Exception as exc:
            st.warning(f"DB unavailable ({exc}), using demo data.")

    # Demo data for when DB is not connected
    dates = pd.date_range(end=datetime.utcnow(), periods=30, freq="D")
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "day_bucket": dates,
        "total_transactions": rng.integers(30_000, 50_000, 30),
        "fraud_transactions": rng.integers(50, 90, 30),
        "fraud_amount": rng.uniform(50_000, 120_000, 30),
        "total_amount": rng.uniform(8_000_000, 15_000_000, 30),
        "fraud_rate_pct": rng.uniform(0.12, 0.22, 30),
        "estimated_loss_prevented": rng.uniform(15_000, 36_000, 30),
        "auto_approved": rng.integers(28_000, 48_000, 30),
        "manual_review": rng.integers(800, 1_500, 30),
        "auto_declined": rng.integers(30, 80, 30),
    })


df = load_data()

# ── Top KPI row ────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    total_tx = int(df["total_transactions"].sum())
    st.metric("Total Transactions", f"{total_tx:,}")
with col2:
    total_fraud = int(df["fraud_transactions"].sum())
    st.metric("Fraud Flagged", f"{total_fraud:,}")
with col3:
    avg_fraud_rate = df["fraud_rate_pct"].mean()
    st.metric("Avg Fraud Rate", f"{avg_fraud_rate:.3f}%")
with col4:
    total_prevented = df["estimated_loss_prevented"].sum()
    st.metric("Est. Loss Prevented (30d)", f"${total_prevented:,.0f}")
with col5:
    total_amount = df["total_amount"].sum()
    st.metric("Total Volume (30d)", f"${total_amount:,.0f}")

st.divider()

# ── Daily trend charts ─────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Daily Transaction Volume vs. Fraud Detections")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["day_bucket"], y=df["total_transactions"],
        name="Total", marker_color="steelblue", opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=df["day_bucket"], y=df["fraud_transactions"],
        name="Fraud", mode="lines+markers",
        yaxis="y2", line=dict(color="crimson", width=2),
    ))
    fig.update_layout(
        yaxis2=dict(overlaying="y", side="right", title="Fraud Count"),
        hovermode="x unified",
        height=350,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Estimated Daily Loss Prevented ($)")
    fig2 = px.area(
        df, x="day_bucket", y="estimated_loss_prevented",
        color_discrete_sequence=["#2ecc71"],
        labels={"estimated_loss_prevented": "$ Prevented", "day_bucket": "Date"},
    )
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

# ── Decision distribution ──────────────────────────────────────────────────────
st.subheader("Decision Distribution (30-day rolling)")
decision_totals = {
    "Auto Approved": int(df["auto_approved"].sum()),
    "Manual Review": int(df["manual_review"].sum()),
    "Auto Declined": int(df["auto_declined"].sum()),
}
fig3 = px.pie(
    names=list(decision_totals.keys()),
    values=list(decision_totals.values()),
    color_discrete_map={
        "Auto Approved": "#2ecc71",
        "Manual Review": "#f39c12",
        "Auto Declined": "#e74c3c",
    },
    hole=0.4,
)
fig3.update_layout(height=300)
st.plotly_chart(fig3, use_container_width=True)
