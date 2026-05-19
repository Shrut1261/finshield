"""
Phase 2: Feature Engineering Validation
=========================================
Validates that all feature modules produce correct output and
analyses correlation of engineered features with fraud labels.

Usage:
    python notebooks/02_feature_engineering.py
"""
# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("data")
PLOTS_DIR = Path("docs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Load a sample and run the feature pipeline

# %%
from ingestion.loaders import DatasetLoader
from features.pipeline import FeaturePipeline, FEATURE_COLUMNS

loader = DatasetLoader(data_dir=DATA_DIR)
df_raw, _ = loader.load_all()

# Use a stratified sample to keep fraud cases for analysis
fraud = df_raw[df_raw["is_fraud"] == 1]
legit = df_raw[df_raw["is_fraud"] == 0].sample(
    n=min(10_000, len(df_raw[df_raw["is_fraud"] == 0])),
    random_state=42,
)
sample = pd.concat([fraud, legit], ignore_index=True)
print(f"Sample: {len(sample):,} rows | fraud_rate={sample['is_fraud'].mean():.4%}")

# %% [markdown]
# ## 2. Run feature pipeline

# %%
pipeline = FeaturePipeline(persist_to_store=False)
df_features = pipeline.run(sample)

present = [c for c in FEATURE_COLUMNS if c in df_features.columns]
print(f"\nFeatures computed: {len(present)}/{len(FEATURE_COLUMNS)}")
missing = [c for c in FEATURE_COLUMNS if c not in df_features.columns]
if missing:
    print(f"Missing (will be 0-filled): {missing[:5]}{'...' if len(missing) > 5 else ''}")

# %% [markdown]
# ## 3. Feature correlation with fraud label

# %%
X = df_features[present].fillna(0)
y = df_features["is_fraud"]

correlations = X.corrwith(y).abs().sort_values(ascending=False)
print("\nTop 15 features by |correlation| with fraud label:")
print(correlations.head(15).round(4).to_string())

fig, ax = plt.subplots(figsize=(10, 6))
correlations.head(20).plot.barh(ax=ax, color="steelblue")
ax.set_xlabel("|Pearson correlation| with is_fraud")
ax.set_title("Top 20 Features by Fraud Correlation")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "feature_correlations.png", dpi=150)
print("\nSaved: docs/plots/feature_correlations.png")
plt.close()

# %% [markdown]
# ## 4. Amount Z-score distribution: fraud vs. legit

# %%
if "amount_zscore" in df_features.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.linspace(-3, 10, 60)
    ax.hist(
        df_features.loc[df_features["is_fraud"]==0, "amount_zscore"].clip(-3, 10),
        bins=bins, alpha=0.6, color="steelblue", label="Legit", density=True,
    )
    ax.hist(
        df_features.loc[df_features["is_fraud"]==1, "amount_zscore"].clip(-3, 10),
        bins=bins, alpha=0.6, color="crimson", label="Fraud", density=True,
    )
    ax.set_xlabel("Amount Z-score (clipped at ±10)")
    ax.set_ylabel("Density")
    ax.set_title("Amount Z-score Distribution: Fraud vs. Legit")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "amount_zscore_dist.png", dpi=150)
    print("Saved: docs/plots/amount_zscore_dist.png")
    plt.close()

print("\nFeature engineering validation complete.")
print(f"Feature matrix shape: {X.shape}")
print(f"Memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
