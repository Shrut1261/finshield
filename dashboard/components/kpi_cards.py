"""Reusable KPI card components for the Streamlit dashboard."""
from __future__ import annotations

import streamlit as st


def metric_row(metrics: list[dict]) -> None:
    """Render a row of st.metric cards.

    Args:
        metrics: List of dicts with keys: label, value, delta (optional),
                 delta_color (optional, default 'normal').

    Example:
        metric_row([
            {"label": "AUC-ROC", "value": "0.978", "delta": "+0.031 vs baseline"},
            {"label": "Recall",  "value": "94.2%", "delta": "+12.3pp"},
        ])
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                delta_color=m.get("delta_color", "normal"),
            )


def fraud_risk_badge(risk_tier: str) -> str:
    """Return a colored emoji badge for a risk tier string."""
    badges = {
        "low": "🟢 LOW",
        "medium": "🟡 MEDIUM",
        "high": "🟠 HIGH",
        "critical": "🔴 CRITICAL",
    }
    return badges.get(risk_tier.lower(), f"⚪ {risk_tier.upper()}")


def decision_badge(decision: str) -> str:
    """Return a human-readable decision badge."""
    badges = {
        "auto_approve": "✅ Auto-Approved",
        "manual_review": "👁 Manual Review",
        "auto_decline": "❌ Auto-Declined",
    }
    return badges.get(decision, decision)
