"""
NeuroMosaic - Exploratory Data Analysis & UMAP Projection
Scales features and visualizes high-dimensional mental state data in 2D.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import umap

PROCESSED_DATA = Path("data/processed/combined_data.csv")
FIGURES_DIR    = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 7)

print("Loading processed dataset...")
df = pd.read_csv(PROCESSED_DATA)
print(f"Dataset shape: {df.shape}")

subject_ids = df['subject_id']
X = df.drop(columns=['subject_id'])

print(f"Number of features: {X.shape[1]}")
print("Feature names (first 10):", list(X.columns[:10]))

print("\n--- Feature Statistics ---")
stats = X.describe().T[['mean', 'std', 'min', 'max']]
print(stats.round(4).to_string())

print("\nScaling features (StandardScaler)...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Scaled data shape: {X_scaled.shape}")
print("Mean of first 5 features after scaling (should be ~0):", X_scaled[:, :5].mean(axis=0).round(4))
print("Std  of first 5 features after scaling (should be ~1):", X_scaled[:, :5].std(axis=0).round(4))

print("\nRunning UMAP to create 2D embedding...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2, verbose=False)
embedding = reducer.fit_transform(X_scaled)
print(f"UMAP embedding shape: {embedding.shape}")

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, c='steelblue', edgecolors='white', s=60)
ax.set_title("NeuroMosaic: UMAP Projection of Subject Mental State Features", fontsize=14, pad=15)
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.grid(True, linestyle='--', alpha=0.3)

fig_path = FIGURES_DIR / "umap_initial.png"
plt.tight_layout()
plt.savefig(fig_path, dpi=150)
print(f"\n[OK] UMAP plot saved to: {fig_path}")

print("\n--- Exploration Complete ---")
print(f"Total subjects: {len(subject_ids)}")
print("If you see clearly separated groups in the plot, the data likely contains distinct clusters.")
print("Next step: Apply unsupervised clustering to discover them automatically.")
