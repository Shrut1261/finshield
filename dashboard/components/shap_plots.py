"""SHAP visualization components for the alert triage page."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def shap_waterfall(
    shap_features: list[dict[str, float]],
    base_value: float = 0.002,
    title: str = "Why this transaction was flagged",
) -> None:
    """Render a SHAP waterfall chart in Streamlit.

    Args:
        shap_features: List of {"feature": str, "shap_value": float} dicts,
                       ordered by absolute importance descending.
        base_value: Model base value (expected fraud probability).
        title: Chart title.
    """
    if not shap_features:
        st.info("No SHAP explanation available for this transaction.")
        return

    features = [s["feature"] for s in shap_features]
    values = [s["shap_value"] for s in shap_features]
    colors = ["crimson" if v > 0 else "steelblue" for v in values]
    labels = [f"+{v:.4f}" if v > 0 else f"{v:.4f}" for v in values]

    fig = go.Figure(go.Bar(
        y=features,
        x=values,
        orientation="h",
        marker_color=colors,
        text=labels,
        textposition="outside",
    ))
    fig.add_vline(x=0, line_width=1, line_color="black")
    fig.update_layout(
        title=title,
        height=max(300, len(features) * 45),
        xaxis_title="SHAP Value → pushes toward fraud (+) or legit (−)",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=80, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Base value (avg fraud probability in training data): {base_value:.4f}. "
        "Red bars increase fraud probability; blue bars decrease it."
    )


def shap_summary_beeswarm(feature_importance: dict[str, float]) -> None:
    """Render a horizontal bar chart as a proxy for the SHAP summary plot."""
    import plotly.express as px
    import pandas as pd

    df = pd.DataFrame(
        list(feature_importance.items()),
        columns=["Feature", "Mean |SHAP|"],
    ).sort_values("Mean |SHAP|")

    fig = px.bar(
        df,
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        color="Mean |SHAP|",
        color_continuous_scale="Reds",
        title="Global Feature Importance (mean |SHAP value|)",
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
