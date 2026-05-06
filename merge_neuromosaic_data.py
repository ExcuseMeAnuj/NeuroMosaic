import pandas as pd
from pathlib import Path

raw_dir       = Path("data/raw")
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

eeg      = pd.read_csv(raw_dir / "eeg_features.csv")
clinical = pd.read_csv(raw_dir / "clinical_scores.csv")

print(f"EEG      -> {eeg.shape}")
print(f"Clinical -> {clinical.shape}")

combined = pd.merge(eeg, clinical, on="subject_id", how="inner")
combined.drop_duplicates(subset="subject_id", inplace=True)

# fill any missing numeric values with column median
ids      = combined[["subject_id"]]
features = combined.drop(columns=["subject_id"])
num_cols = features.select_dtypes(include="number").columns
features[num_cols] = features[num_cols].fillna(features[num_cols].median())

combined_clean = pd.concat([ids, features], axis=1)
combined_clean.to_csv(processed_dir / "combined_data.csv", index=False)

print(f"Merged   -> {combined_clean.shape}")
print(f"Saved to data/processed/combined_data.csv")
