"""Drift Monitoring — weekly PSI scores per feature."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Drift Monitoring", page_icon="📈", layout="wide")
st.title("📈 Model Drift Monitoring")
st.markdown(
    "Population Stability Index (PSI) measures feature distribution shift from training to production. "
    "**PSI < 0.10** = stable | **0.10–0.20** = monitor | **> 0.20** = retrain."
)


@st.cache_data(ttl=600)
def load_drift_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    features = [
        "amount", "tx_count_customer_id_24h", "amount_zscore",
        "hour_of_day", "day_of_week", "tx_sum_card_id_7d",
        "is_new_country", "merchant_fraud_rate", "device_id_card_count",
        "time_since_last_tx_hours",
    ]
    weeks = pd.date_range(end=pd.Timestamp.today(), periods=8, freq="W")
    rows = []
    for feat in features:
        base_psi = rng.uniform(0.02, 0.08)
        for i, week in enumerate(weeks):
            drift = base_psi + rng.uniform(-0.01, 0.03) * i
            psi = max(0, min(0.35, drift))
            rows.append({
                "feature": feat,
                "week_start": week,
                "psi_score": round(float(psi), 5),
                "status": "stable" if psi < 0.10 else ("monitor" if psi < 0.20 else "retrain"),
            })
    return pd.DataFrame(rows)


df = load_drift_data()
latest_week = df["week_start"].max()
latest = df[df["week_start"] == latest_week].copy()

# ── Current week status ───────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    stable = (latest["status"] == "stable").sum()
    st.metric("✅ Stable Features", stable)
with col2:
    monitor = (latest["status"] == "monitor").sum()
    st.metric("⚠️ Monitor", monitor, delta_color="off")
with col3:
    retrain = (latest["status"] == "retrain").sum()
    st.metric("🔴 Retrain Alert", retrain, delta_color="inverse")

st.divider()

# ── PSI bar chart ─────────────────────────────────────────────────────────────
st.subheader(f"PSI Scores — Week of {latest_week.strftime('%Y-%m-%d')}")
color_map = {"stable": "#2ecc71", "monitor": "#f39c12", "retrain": "#e74c3c"}
fig = px.bar(
    latest.sort_values("psi_score", ascending=True),
    x="psi_score",
    y="feature",
    orientation="h",
    color="status",
    color_discrete_map=color_map,
    labels={"psi_score": "PSI Score", "feature": "Feature"},
)
fig.add_vline(x=0.10, line_dash="dash", line_color="#f39c12", annotation_text="Monitor threshold")
fig.add_vline(x=0.20, line_dash="dash", line_color="#e74c3c", annotation_text="Retrain threshold")
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# ── PSI trend over time ────────────────────────────────────────────────────────
st.subheader("PSI Trend — Last 8 Weeks")
selected_feature = st.selectbox("Select feature", df["feature"].unique())
feat_df = df[df["feature"] == selected_feature]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=feat_df["week_start"], y=feat_df["psi_score"],
    mode="lines+markers", line=dict(color="#007bff", width=2),
    name="PSI Score",
))
fig2.add_hrect(y0=0, y1=0.10, fillcolor="#2ecc71", opacity=0.1, annotation_text="Stable")
fig2.add_hrect(y0=0.10, y1=0.20, fillcolor="#f39c12", opacity=0.1, annotation_text="Monitor")
fig2.add_hrect(y0=0.20, y1=0.40, fillcolor="#e74c3c", opacity=0.1, annotation_text="Retrain")
fig2.update_layout(height=350, xaxis_title="Week", yaxis_title="PSI Score")
st.plotly_chart(fig2, use_container_width=True)
