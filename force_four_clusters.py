import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap

data_path   = Path("data/processed/combined_data.csv")
figures_dir = Path("results/figures")
results_dir = Path("results")
figures_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)
print(f"Loaded {df.shape[0]} subjects with {df.shape[1]-1} features")

# keep only numeric, drop NaN columns, fill remaining gaps
X = df.drop(columns=["subject_id"]).select_dtypes(include="number")
X = X.dropna(axis=1).fillna(X.median())

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# UMAP for the 2D picture
print("Running UMAP...")
reducer  = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_scaled)

# GMM — forced k=4 (we want exactly 4 clusters)
print("Fitting GMM with k=4...")
best_k   = 4
best_gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
best_gmm.fit(X_scaled)
labels   = best_gmm.predict(X_scaled)

sil = silhouette_score(X_scaled, labels)
dbi = davies_bouldin_score(X_scaled, labels)
print(f"\nBest k={best_k}  |  Silhouette={sil:.4f}  |  Davies-Bouldin={dbi:.4f}")

# save labelled data
out = df.copy()
out["cluster"] = labels
out.to_csv(results_dir / "clustered_subjects_4.csv", index=False)

# cluster profile
profile_cols = ["alpha_asymmetry", "theta_beta_ratio", "PHQ9_total", "GAD7_total"]
available    = [c for c in profile_cols if c in out.columns]
print("\nCluster profiles (mean):")
print(out.groupby("cluster")[available].mean().round(2).to_string())

# UMAP scatter plot
fig, ax = plt.subplots(figsize=(11, 7))
palette = sns.color_palette("husl", len(np.unique(labels)))
for i, label in enumerate(np.unique(labels)):
    mask = labels == label
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               color=palette[i], edgecolors="white", s=55, alpha=0.8,
               label=f"Cluster {label}")

ax.set_title(f"NeuroMosaic — GMM Clusters (k={best_k})", fontsize=14)
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.legend(title="Cluster")
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / "clusters_4_force.png", dpi=150)
print(f"\nPlot saved -> results/figures/clusters_4_force.png")
