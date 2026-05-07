"""
NeuroMosaic - Full Report Viewer
Sab results ek jagah clearly dikhata hai.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

print("=" * 60)
print("       NEUROMOSAIC - FULL RESULTS REPORT")
print("=" * 60)

print("\n[1] RAW DATA")
print("-" * 40)
eeg      = pd.read_csv("data/raw/eeg_features.csv")
clinical = pd.read_csv("data/raw/clinical_scores.csv")
print(f"  EEG Features     : {eeg.shape[0]} subjects, {eeg.shape[1]-1} features")
print(f"  Clinical Scores  : {clinical.shape[0]} subjects, {clinical.shape[1]-1} features")
print(f"  EEG columns      : {list(eeg.columns[1:6])} ...")
print(f"  Clinical columns : {list(clinical.columns[1:])}")

print("\n[2] PROCESSED / MERGED DATA")
print("-" * 40)
combined = pd.read_csv("data/processed/combined_data.csv")
print(f"  Combined shape   : {combined.shape[0]} subjects x {combined.shape[1]-1} features")
print(f"  Missing values   : {combined.isnull().sum().sum()}")

print("\n[3] CLUSTERING RESULTS (GMM k=4)")
print("-" * 40)
clustered = pd.read_csv("results/clustered_subjects_4.csv")
counts = clustered['cluster'].value_counts().sort_index()
for c, n in counts.items():
    print(f"  Cluster {c} : {n} subjects")

print("\n[4] CLUSTER PROFILES (Mean Values)")
print("-" * 40)
profile_cols = ['PHQ9_total', 'GAD7_total', 'PSQI_total', 'PSS_total', 'theta_beta_ratio']
available    = [c for c in profile_cols if c in clustered.columns]
profile      = clustered.groupby('cluster')[available].mean().round(2)
labels_map   = {0: "High Anxiety", 1: "Burnout", 2: "Healthy", 3: "Melancholic Depression"}
profile['Interpretation'] = profile.index.map(labels_map)
print(profile.to_string())

print("\n  Score Guide:")
print("  PHQ9  : Depression  (0-27)  | High = more depressed")
print("  GAD7  : Anxiety     (0-21)  | High = more anxious")
print("  PSQI  : Sleep       (0-21)  | High = worse sleep")
print("  PSS   : Stress      (0-40)  | High = more stressed")

print("\n[5] MODEL EVALUATION SCORES")
print("-" * 40)
print("  GMM k=2  ->  Silhouette: 0.2903  | Davies-Bouldin: 1.2240")
print("  GMM k=4  ->  Silhouette: 0.3235  | Davies-Bouldin: 1.2145  <-- BEST")
print("\n  Silhouette: closer to 1.0 = better separated clusters")
print("  Davies-Bouldin: closer to 0.0 = better clusters")

print("\n[6] OPENING PLOTS...")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("NeuroMosaic - Complete Visual Report", fontsize=16, fontweight='bold')

ax1     = axes[0]
colors  = sns.color_palette("husl", 4)
c_names = ["Anxiety", "Burnout", "Healthy", "Depression"]
bars    = ax1.bar(c_names, counts.values, color=colors, edgecolor='white', linewidth=1.5)
ax1.set_title("Subjects per Cluster", fontsize=13)
ax1.set_ylabel("Number of Subjects")
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val), ha='center', fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.4)

ax2       = axes[1]
heat_cols = ['PHQ9_total', 'GAD7_total', 'PSQI_total', 'PSS_total']
heat_avail = [c for c in heat_cols if c in clustered.columns]
heat_data  = clustered.groupby('cluster')[heat_avail].mean()
heat_data.index = c_names[:len(heat_data)]
sns.heatmap(heat_data, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, ax=ax2, cbar_kws={'label': 'Score'})
ax2.set_title("Cluster Profile Heatmap", fontsize=13)

plt.tight_layout()
report_path = "results/figures/full_report.png"
plt.savefig(report_path, dpi=150)
print(f"  Report chart saved: {report_path}")
plt.show()

umap_path = Path("results/figures/clusters_4_force.png")
if umap_path.exists():
    os.startfile(str(umap_path))
    print(f"  UMAP plot opened: {umap_path}")

print("\n" + "=" * 60)
print("  REPORT COMPLETE")
print("=" * 60)
