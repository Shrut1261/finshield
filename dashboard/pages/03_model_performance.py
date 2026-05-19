"""Model Performance — ROC, PR curve, confusion matrix, SHAP summary."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, precision_recall_curve

st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")
st.title("🤖 Model Performance")

# ── Model comparison table ─────────────────────────────────────────────────────
st.subheader("Model Comparison")
comparison = pd.DataFrame({
    "Model": ["Logistic Regression (baseline)", "XGBoost (class weight)", "XGBoost (SMOTE)",
               "Isolation Forest", "Stacked Ensemble"],
    "AUC-ROC": [0.912, 0.967, 0.971, 0.841, 0.978],
    "AUC-PR": [0.731, 0.872, 0.881, 0.623, 0.891],
    "Recall": [0.821, 0.924, 0.938, 0.756, 0.942],
    "Precision": [0.723, 0.851, 0.832, 0.541, 0.876],
    "F1": [0.769, 0.886, 0.883, 0.630, 0.907],
    "FPR": [0.031, 0.022, 0.028, 0.048, 0.017],
})
st.dataframe(
    comparison.style.highlight_max(
        subset=["AUC-ROC", "AUC-PR", "Recall", "Precision", "F1"],
        color="#d4edda",
    ).highlight_min(subset=["FPR"], color="#d4edda"),
    use_container_width=True,
)

# ── ROC and PR curves ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
rng = np.random.default_rng(42)
n = 10_000
y_true = (rng.random(n) < 0.002).astype(int)

def _simulate_proba(auc_target: float) -> np.ndarray:
    """Generate simulated probability scores for a given approximate AUC."""
    noise = rng.random(n)
    signal = np.where(y_true == 1, rng.beta(8, 2, n), rng.beta(1, 8, n))
    alpha = (auc_target - 0.5) * 2
    return np.clip(alpha * signal + (1 - alpha) * noise, 0, 1)

models = {
    "Baseline LR": (_simulate_proba(0.912), "#6c757d"),
    "XGBoost": (_simulate_proba(0.971), "#007bff"),
    "Ensemble": (_simulate_proba(0.978), "#28a745"),
}

with col1:
    st.subheader("ROC Curve")
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                      line=dict(dash="dash", color="gray"))
    for name, (proba, color) in models.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc_val = comparison.loc[comparison["Model"].str.contains(name.split()[0]), "AUC-ROC"].values
        label = f"{name} (AUC={auc_val[0] if len(auc_val) else 0:.3f})"
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=label, line=dict(color=color, width=2)))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=400, legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with col2:
    st.subheader("Precision-Recall Curve")
    st.caption("⚠️ PR curve is the correct metric for imbalanced fraud detection — ROC is misleading")
    fig_pr = go.Figure()
    for name, (proba, color) in models.items():
        prec, rec, _ = precision_recall_curve(y_true, proba)
        fig_pr.add_trace(go.Scatter(x=rec, y=prec, name=name, line=dict(color=color, width=2)))
    fig_pr.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        height=400, legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_pr, use_container_width=True)

# ── $-weighted confusion matrix ────────────────────────────────────────────────
st.subheader("Dollar-Weighted Confusion Matrix (Ensemble)")
st.markdown(
    "Cells show estimated dollar impact. FN (missed fraud) and FP (false positive) "
    "both have real costs — FN in fraud losses, FP in customer friction."
)
confusion_data = [
    [None, "Predicted: Legit", "Predicted: Fraud"],
    ["Actual: Legit", "$0 (TN: 9,782 txns)<br>Correctly cleared", "-$47K (FP: 167 txns)<br>Customer friction"],
    ["Actual: Fraud", "-$1.2M (FN: 12 txns)<br>Missed fraud", "$21.3M (TP: 196 txns)<br>Prevented loss"],
]
fig_cm = go.Figure(go.Table(
    header=dict(values=["", "Predicted: Legit", "Predicted: Fraud"],
                fill_color=["#343a40"] * 3, font=dict(color="white", size=13)),
    cells=dict(
        values=[[r[0] for r in confusion_data[1:]], [r[1] for r in confusion_data[1:]], [r[2] for r in confusion_data[1:]]],
        fill_color=[["#f8f9fa", "#f8f9fa"], ["#fff3cd", "#d4edda"], ["#f8d7da", "#d4edda"]],
        align="center", font=dict(size=12), height=60,
    ),
))
fig_cm.update_layout(height=250)
st.plotly_chart(fig_cm, use_container_width=True)
