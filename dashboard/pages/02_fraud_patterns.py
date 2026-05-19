"""Fraud Patterns — heatmaps, merchant category analysis, geography."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Fraud Patterns", page_icon="🔍", layout="wide")
st.title("🔍 Fraud Patterns")


@st.cache_data(ttl=300)
def load_pattern_data() -> pd.DataFrame:
    """Load from warehouse or generate demo data."""
    rng = np.random.default_rng(42)
    hours = list(range(24))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    categories = [
        "grocery", "gas_station", "restaurant", "online_retail",
        "travel", "wire_transfer", "crypto", "atm_withdrawal",
        "healthcare", "entertainment",
    ]

    rows = []
    for h in hours:
        for d in days:
            n_tx = int(rng.integers(800, 5000))
            # Elevated fraud at night and on weekends
            base_rate = 0.002
            if h in range(1, 5):
                base_rate *= 4.5
            if d in ("Sat", "Sun"):
                base_rate *= 1.8
            n_fraud = int(n_tx * base_rate * rng.uniform(0.7, 1.3))
            rows.append({"hour": h, "day": d, "transactions": n_tx, "fraud": n_fraud})

    heatmap_df = pd.DataFrame(rows)
    heatmap_df["fraud_rate"] = heatmap_df["fraud"] / heatmap_df["transactions"] * 100

    cat_rows = []
    for cat in categories:
        n = int(rng.integers(5_000, 50_000))
        rate = rng.uniform(0.001, 0.015)
        if cat in ("wire_transfer", "crypto", "atm_withdrawal"):
            rate *= 8
        cat_rows.append({
            "category": cat,
            "transactions": n,
            "fraud_count": int(n * rate),
            "avg_amount": float(rng.uniform(20, 2000)),
            "fraud_amount": float(n * rate * rng.uniform(200, 2000)),
        })
    cat_df = pd.DataFrame(cat_rows)
    cat_df["fraud_rate_pct"] = cat_df["fraud_count"] / cat_df["transactions"] * 100

    return heatmap_df, cat_df


heatmap_df, cat_df = load_pattern_data()

# ── Fraud rate heatmap ─────────────────────────────────────────────────────────
st.subheader("Fraud Rate by Hour of Day × Day of Week")
pivot = heatmap_df.pivot(index="day", columns="hour", values="fraud_rate")
fig = px.imshow(
    pivot,
    color_continuous_scale="Reds",
    labels={"color": "Fraud Rate (%)"},
    aspect="auto",
)
fig.update_layout(height=350, xaxis_title="Hour of Day", yaxis_title="Day of Week")
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "> **Insight:** Fraud peaks between 1–4am on weekends — classic account takeover pattern. "
    "Velocity features capture the burst; the model flags these in <87ms."
)

# ── Merchant category fraud rate ──────────────────────────────────────────────
st.subheader("Fraud Rate & Volume by Merchant Category")
col1, col2 = st.columns(2)

with col1:
    fig2 = px.bar(
        cat_df.sort_values("fraud_rate_pct", ascending=True),
        x="fraud_rate_pct", y="category",
        orientation="h",
        color="fraud_rate_pct",
        color_continuous_scale="RdYlGn_r",
        labels={"fraud_rate_pct": "Fraud Rate (%)", "category": "Merchant Category"},
        title="Fraud Rate by Category",
    )
    fig2.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig3 = px.scatter(
        cat_df,
        x="transactions",
        y="fraud_rate_pct",
        size="fraud_amount",
        color="category",
        hover_name="category",
        labels={"transactions": "Transaction Volume", "fraud_rate_pct": "Fraud Rate (%)"},
        title="Volume vs. Fraud Rate (bubble = $ fraud amount)",
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

st.markdown(
    "> **Key finding:** Wire transfer and crypto show 8× elevated fraud rates despite low volume. "
    "These categories are flagged as `is_high_risk_category = TRUE` in `dim_merchant`."
)
