# Step 1 — Save Trained Model
# Run this ONCE now to save the scaler + GMM model from current Kaggle pipeline.
# Jab bhi EEG machine aaye, ye saved model use hoga — retraining ki zaroorat nahi.

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

models_dir = Path("future_eeg/models")
models_dir.mkdir(parents=True, exist_ok=True)

# load the same data current pipeline uses
df = pd.read_csv("data/processed/combined_data.csv")

X = df.drop(columns=["subject_id"]).select_dtypes(include="number")
X = X.dropna(axis=1).fillna(X.median())

feature_cols = X.columns.tolist()

# train scaler and GMM — same settings as force_four_clusters.py
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
gmm.fit(X_scaled)

# save everything
joblib.dump(scaler,       models_dir / "scaler.pkl")
joblib.dump(gmm,          models_dir / "gmm_model.pkl")
joblib.dump(feature_cols, models_dir / "feature_order.pkl")

print(f"[OK] scaler.pkl       saved")
print(f"[OK] gmm_model.pkl    saved")
print(f"[OK] feature_order.pkl saved  ({len(feature_cols)} features)")
print(f"\nFirst 10 features model expects:")
for i, f in enumerate(feature_cols[:10]):
    print(f"  {i+1}. {f}")
print(f"\nModel is ready. Jab EEG machine aaye, live_predict.py chalao.")
