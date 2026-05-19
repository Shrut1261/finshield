"""
Phase 2: Exploratory Data Analysis
===================================
Run as a script or open as a Jupyter notebook with Jupytext.

Key questions answered:
  1. What does the class imbalance look like across datasets?
  2. What does the amount distribution look like for fraud vs. legit?
  3. What time-of-day and day-of-week patterns exist?
  4. Which features correlate most strongly with fraud?

Usage:
    python notebooks/01_eda.py
    # or in Jupyter: pip install jupytext && jupytext --to notebook notebooks/01_eda.py
"""
# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script mode

DATA_DIR = Path("data")
PLOTS_DIR = Path("docs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Load All Datasets

# %%
from ingestion.loaders import DatasetLoader

loader = DatasetLoader(data_dir=DATA_DIR)
df, stats = loader.load_all()

print(f"Combined: {len(df):,} rows")
print(f"Overall fraud rate: {df['is_fraud'].mean():.4%}")
print()
for s in stats:
    print(f"  {s.dataset:<20} {s.rows_loaded:>8,} rows  fraud={s.fraud_rate:.4%}")

# %% [markdown]
# ## 2. Class Imbalance — Per Dataset

# %%
fig, axes = plt.subplots(1, len(stats), figsize=(4 * len(stats), 4))
if len(stats) == 1:
    axes = [axes]

for ax, s in zip(axes, stats):
    counts = [s.rows_loaded - s.fraud_count, s.fraud_count]
    ax.bar(["Legit", "Fraud"], counts, color=["steelblue", "crimson"])
    ax.set_title(f"{s.dataset}\nFraud rate: {s.fraud_rate:.4%}")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts) * 0.01, f"{v:,}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "class_imbalance.png", dpi=150)
print("Saved: docs/plots/class_imbalance.png")
plt.close()

# %% [markdown]
# ## 3. Amount Distribution: Fraud vs. Legit

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

legit_amounts = df.loc[df["is_fraud"] == 0, "amount"].clip(upper=1000)
fraud_amounts = df.loc[df["is_fraud"] == 1, "amount"].clip(upper=1000)

ax1.hist(legit_amounts, bins=80, color="steelblue", alpha=0.7, label="Legit")
ax1.hist(fraud_amounts, bins=80, color="crimson", alpha=0.7, label="Fraud")
ax1.set_xlabel("Amount (clipped at $1,000)")
ax1.set_ylabel("Count")
ax1.set_title("Amount Distribution")
ax1.legend()

# Log scale for better visibility
ax2.hist(legit_amounts + 0.01, bins=80, color="steelblue", alpha=0.7, label="Legit")
ax2.hist(fraud_amounts + 0.01, bins=80, color="crimson", alpha=0.7, label="Fraud")
ax2.set_xlabel("Amount (log scale)")
ax2.set_yscale("log")
ax2.set_title("Amount Distribution (log y-axis)")
ax2.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "amount_distribution.png", dpi=150)
print("Saved: docs/plots/amount_distribution.png")
plt.close()

# Key stats
print("\nAmount statistics:")
print(pd.DataFrame({
    "Legit": df.loc[df["is_fraud"] == 0, "amount"].describe(),
    "Fraud": df.loc[df["is_fraud"] == 1, "amount"].describe(),
}).round(2))

# %% [markdown]
# ## 4. Fraud Rate by Hour of Day

# %%
if "raw_timestamp" in df.columns:
    df["hour"] = pd.to_datetime(df["raw_timestamp"]).dt.hour
    hourly = df.groupby("hour")["is_fraud"].agg(["sum", "count"])
    hourly["rate"] = hourly["sum"] / hourly["count"]

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(hourly.index, hourly["rate"] * 100, color="steelblue")
    # Highlight high-fraud hours
    for bar, h in zip(bars, hourly.index):
        if hourly.loc[h, "rate"] > hourly["rate"].mean() * 1.5:
            bar.set_color("crimson")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate by Hour of Day\n(red = significantly above average)")
    ax.set_xticks(range(24))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fraud_by_hour.png", dpi=150)
    print("Saved: docs/plots/fraud_by_hour.png")
    plt.close()

# %% [markdown]
# ## 5. Summary Statistics

# %%
summary = {
    "Total transactions": f"{len(df):,}",
    "Fraud transactions": f"{df['is_fraud'].sum():,}",
    "Fraud rate": f"{df['is_fraud'].mean():.4%}",
    "Median legit amount": f"${df.loc[df['is_fraud']==0, 'amount'].median():.2f}",
    "Median fraud amount": f"${df.loc[df['is_fraud']==1, 'amount'].median():.2f}",
    "Datasets": len(stats),
}
print("\n── EDA Summary ─────────────────────────────")
for k, v in summary.items():
    print(f"  {k:<30} {v}")
print("────────────────────────────────────────────")
print("\nEDA complete. See docs/plots/ for visualizations.")
