import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/processed/combined_data.csv")
X  = df.drop(columns=["subject_id"]).select_dtypes(include="number")
X  = X.dropna(axis=1).fillna(X.median())
X_scaled = StandardScaler().fit_transform(X)

ks   = range(2, 9)
bics = []
sils = []

print("Testing k = 2 to 8...\n")
for k in ks:
    gmm    = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    bic    = gmm.bic(X_scaled)
    sil    = silhouette_score(X_scaled, labels)
    bics.append(bic)
    sils.append(sil)
    print(f"  k={k}  BIC={bic:.1f}  Silhouette={sil:.4f}")

optimal_k   = list(ks)[int(np.argmin(bics))]
optimal_sil = sils[optimal_k - 2]

print(f"\nOptimal k by BIC : {optimal_k}")
print(f"Silhouette at k={optimal_k} : {optimal_sil:.4f}")

# plot
fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(ks, bics, "bo-", linewidth=2, markersize=7, label="BIC")
ax1.axvline(x=optimal_k, color="gray", linestyle="--", alpha=0.5, label=f"Optimal k={optimal_k}")
ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
ax1.set_ylabel("BIC  (lower = better)", color="b", fontsize=11)
ax1.tick_params(axis="y", labelcolor="b")

ax2 = ax1.twinx()
ax2.plot(ks, sils, "rs--", linewidth=2, markersize=7, label="Silhouette")
ax2.set_ylabel("Silhouette Score  (higher = better)", color="r", fontsize=11)
ax2.tick_params(axis="y", labelcolor="r")

plt.title("NeuroMosaic — Optimal Cluster Selection (BIC + Silhouette)", fontsize=13)
fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
plt.tight_layout()

out = Path("results/figures/optimal_k.png")
plt.savefig(out, dpi=150)
print(f"\nPlot saved -> {out}")
plt.show()
