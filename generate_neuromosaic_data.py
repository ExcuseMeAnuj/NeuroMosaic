"""
NeuroMosaic - Synthetic Multi-Modal Dataset Generator
Creates realistic EEG + Clinical features with hidden clusters.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N_SUBJECTS = 200

archetypes = np.random.randint(0, 4, N_SUBJECTS)

def generate_eeg_features(n, archetype):
    delta_base = 0.5 + np.random.normal(0, 0.1, n)
    theta_base = 0.4 + np.random.normal(0, 0.1, n)
    alpha_base = 0.6 + np.random.normal(0, 0.1, n)
    beta_base  = 0.3 + np.random.normal(0, 0.05, n)
    gamma_base = 0.2 + np.random.normal(0, 0.05, n)

    if archetype == 0:
        beta_base  += 0.3
        gamma_base += 0.1
        alpha_base -= 0.1
    elif archetype == 1:
        theta_base += 0.3
        alpha_base -= 0.2
        beta_base  -= 0.05
    elif archetype == 2:
        pass
    elif archetype == 3:
        beta_base  -= 0.2
        alpha_base += 0.2
        theta_base += 0.1
        gamma_base -= 0.1

    channels = ['F3', 'F4', 'C3', 'C4', 'Pz']
    bands    = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    data = {}
    for ch in channels:
        for band in bands:
            if band == 'delta':
                val = delta_base + np.random.normal(0, 0.02, n)
            elif band == 'theta':
                val = theta_base + np.random.normal(0, 0.02, n)
            elif band == 'alpha':
                val = alpha_base + np.random.normal(0, 0.03, n)
            elif band == 'beta':
                val = beta_base  + np.random.normal(0, 0.02, n)
            else:
                val = gamma_base + np.random.normal(0, 0.01, n)
            data[f"{band}_{ch}"] = np.clip(val, 0, 1)

    df_eeg = pd.DataFrame(data)
    df_eeg['alpha_asymmetry']  = df_eeg['alpha_F4'] - df_eeg['alpha_F3']
    df_eeg['theta_beta_ratio'] = ((df_eeg['theta_F3'] + df_eeg['theta_F4'])/2) / ((df_eeg['beta_F3'] + df_eeg['beta_F4'])/2)
    df_eeg.insert(0, 'subject_id', [f'S{i:03d}' for i in range(n)])
    return df_eeg

eeg_data = pd.concat([generate_eeg_features(1, a) for a in archetypes], ignore_index=True)
eeg_data['subject_id'] = [f'S{i:03d}' for i in range(N_SUBJECTS)]

def generate_clinical_scores(n, archetype):
    scores = pd.DataFrame()
    scores['subject_id'] = [f'S{i:03d}' for i in range(n)]

    if archetype == 1:
        phq9 = np.random.normal(22, 3, n)
    elif archetype == 3:
        phq9 = np.random.normal(15, 4, n)
    else:
        phq9 = np.random.normal(5, 2, n)

    if archetype == 0:
        gad7 = np.random.normal(18, 3, n)
    else:
        gad7 = np.random.normal(4, 2, n)

    if archetype == 3:
        psqi = np.random.normal(16, 2, n)
    elif archetype == 1:
        psqi = np.random.normal(14, 3, n)
    else:
        psqi = np.random.normal(5, 2, n)

    if archetype == 0:
        pss = np.random.normal(30, 3, n)
    elif archetype == 1:
        pss = np.random.normal(28, 4, n)
    elif archetype == 3:
        pss = np.random.normal(25, 5, n)
    else:
        pss = np.random.normal(10, 3, n)

    scores['PHQ9_total'] = np.clip(np.round(phq9), 0, 27)
    scores['GAD7_total'] = np.clip(np.round(gad7), 0, 21)
    scores['PSQI_total'] = np.clip(np.round(psqi), 0, 21)
    scores['PSS_total']  = np.clip(np.round(pss),  0, 40)
    return scores

clinical_data = pd.concat([generate_clinical_scores(1, a) for a in archetypes], ignore_index=True)
clinical_data['subject_id'] = [f'S{i:03d}' for i in range(N_SUBJECTS)]

out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)
eeg_data.to_csv(out_dir / "eeg_features.csv", index=False)
clinical_data.to_csv(out_dir / "clinical_scores.csv", index=False)

print(f"[OK] Synthetic NeuroMosaic dataset created: {N_SUBJECTS} subjects")
print(f"   EEG features saved: {out_dir / 'eeg_features.csv'}")
print(f"   Clinical scores saved: {out_dir / 'clinical_scores.csv'}")
print("   Hidden archetypes: 0=Anxious, 1=Depressed, 2=Healthy, 3=Burnout (for validation only).")
