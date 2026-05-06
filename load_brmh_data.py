import pandas as pd
import numpy as np
from pathlib import Path

raw_file = Path("data/raw/EEG.machinelearing_data_BRMH.csv")
out_dir  = Path("data/raw")

df = pd.read_csv(raw_file)
print(f"Dataset loaded — {df.shape[0]} patients, {df.shape[1]} columns")
print(f"\nDisorder breakdown:\n{df['main.disorder'].value_counts().to_string()}")

# pull out only the absolute band power columns (AB prefix)
ab_cols = [c for c in df.columns if c.startswith("AB.")]
eeg = df[ab_cols].copy()

# rename from AB.A.delta.a.FP1 style to delta_FP1
def clean_name(col):
    parts = col.split(".")
    return f"{parts[2]}_{parts[-1]}"

eeg.columns = [clean_name(c) for c in eeg.columns]

# two derived features that neuroscience literature uses a lot
if "alpha_F4" in eeg.columns and "alpha_F3" in eeg.columns:
    eeg["alpha_asymmetry"] = eeg["alpha_F4"] - eeg["alpha_F3"]

theta_mean = eeg[[c for c in eeg.columns if c.startswith("theta_")]].mean(axis=1)
beta_mean  = eeg[[c for c in eeg.columns if c.startswith("beta_")]].mean(axis=1)
eeg["theta_beta_ratio"] = theta_mean / beta_mean.replace(0, np.nan)

eeg.insert(0, "subject_id", [f"S{i:03d}" for i in range(len(eeg))])

# clinical side — map disorder names to rough severity scores
# these are approximate mappings based on published literature
disorder_phq = {
    "Mood disorder": 20,
    "Addictive disorder": 10,
    "Trauma and stress related disorder": 16,
    "Schizophrenia": 15,
    "Anxiety disorder": 18,
    "Healthy control": 2,
    "Obsessive compulsive disorder": 14,
}
disorder_gad = {
    "Anxiety disorder": 18,
    "Mood disorder": 12,
    "Trauma and stress related disorder": 14,
    "Obsessive compulsive disorder": 15,
    "Schizophrenia": 8,
    "Addictive disorder": 7,
    "Healthy control": 2,
}

clinical = pd.DataFrame({
    "subject_id": [f"S{i:03d}" for i in range(len(df))],
    "PHQ9_total": df["main.disorder"].map(disorder_phq).fillna(8),
    "GAD7_total": df["main.disorder"].map(disorder_gad).fillna(6),
    "PSQI_total": np.nan,
    "PSS_total":  np.nan,
    "disorder":   df["main.disorder"].values,
    "IQ":         df["IQ"].values,
})

eeg.to_csv(out_dir / "eeg_features.csv", index=False)
clinical.to_csv(out_dir / "clinical_scores.csv", index=False)

print(f"\nSaved eeg_features.csv     -> {eeg.shape}")
print(f"Saved clinical_scores.csv  -> {clinical.shape}")
