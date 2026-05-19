"""Alert Triage Queue — live scoring and SHAP waterfall explanations."""
from __future__ import annotations

import os
import time
from datetime import datetime

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Alert Triage", page_icon="🚨", layout="wide")
st.title("🚨 Alert Triage Queue")
st.caption("Manual review queue — high and critical risk transactions awaiting analyst decision")


# ── Demo alert queue ──────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def generate_alert_queue() -> pd.DataFrame:
    rng = np.random.default_rng(int(time.time()) % 1000)
    alerts = []
    scenarios = ["after_hours_large", "geo_anomaly", "account_takeover", "card_testing", "unknown"]
    categories = ["wire_transfer", "crypto", "atm_withdrawal", "online_retail", "travel"]
    for i in range(20):
        prob = rng.uniform(0.60, 0.99)
        tier = "critical" if prob >= 0.85 else "high"
        alerts.append({
            "transaction_id": f"ALERT_{rng.integers(100000, 999999)}",
            "customer_id": f"CUST_{rng.integers(1000, 9999):04d}",
            "amount": round(float(rng.uniform(200, 8000)), 2),
            "merchant_category": rng.choice(categories),
            "country": rng.choice(["US", "NG", "RO", "CN", "BR"]),
            "fraud_probability": round(float(prob), 4),
            "risk_tier": tier,
            "top_reason": rng.choice(scenarios),
            "flagged_at": (datetime.utcnow()).strftime("%H:%M:%S"),
        })
    return pd.DataFrame(alerts).sort_values("fraud_probability", ascending=False)


alerts_df = generate_alert_queue()

# ── Queue summary ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Alerts", len(alerts_df))
with col2:
    critical = (alerts_df["risk_tier"] == "critical").sum()
    st.metric("Critical", critical, delta_color="inverse")
with col3:
    total_exposure = alerts_df["amount"].sum()
    st.metric("Total Exposure", f"${total_exposure:,.0f}")
with col4:
    avg_prob = alerts_df["fraud_probability"].mean()
    st.metric("Avg Fraud Prob", f"{avg_prob:.1%}")

st.divider()

# ── Alert table ───────────────────────────────────────────────────────────────
st.subheader("Alert Queue")
selected = st.dataframe(
    alerts_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "fraud_probability": st.column_config.ProgressColumn(
            "Fraud Probability", min_value=0, max_value=1, format="%.1%"
        ),
        "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
        "risk_tier": st.column_config.TextColumn("Risk Tier"),
    },
    on_select="rerun",
    selection_mode="single-row",
)

# ── SHAP waterfall for selected alert ────────────────────────────────────────
st.subheader("SHAP Feature Contribution — Selected Alert")
st.caption("SHAP (SHapley Additive exPlanations) shows *why* the model flagged this transaction.")

# Simulated SHAP values for demo
shap_features = [
    ("tx_sum_customer_id_1h", 1.82, "crimson"),
    ("amount_zscore", 1.41, "crimson"),
    ("is_new_country", 0.93, "crimson"),
    ("hour_of_day", 0.71, "crimson"),
    ("merchant_category_crypto", 0.65, "crimson"),
    ("tx_count_card_id_24h", -0.31, "steelblue"),
    ("days_since_last_tx", -0.18, "steelblue"),
]

features = [f[0] for f in shap_features]
values = [f[1] for f in shap_features]
colors = [f[2] for f in shap_features]

fig = go.Figure(go.Bar(
    y=features,
    x=values,
    orientation="h",
    marker_color=colors,
    text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in values],
    textposition="outside",
))
fig.add_vline(x=0, line_width=1, line_color="black")
fig.update_layout(
    height=350,
    xaxis_title="SHAP Value (impact on fraud probability)",
    yaxis_title="Feature",
    title="Why this transaction was flagged",
    yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Reading the chart:** Red bars push the probability toward fraud; blue bars push toward legitimate.
The velocity feature `tx_sum_customer_id_1h` (total spend in the last hour) is the biggest signal here —
this card spent $3,200 in a single hour, well above its 90-day average of $120.
""")
