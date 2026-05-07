# Live EEG Prediction — NeuroMosaic
# Jab EEG machine aaye tab chalana.
# Abhi SYNTHETIC_BOARD se test kar sakte ho bina machine ke.
#
# Supported devices (brainflow compatible):
#   Muse 2 / Muse S     — sabse sasta, ~20-25k
#   OpenBCI Cyton       — zyada channels, research grade
#   Synthetic Board     — fake data, testing ke liye, koi hardware nahi chahiye
#
# Install: pip install brainflow

import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.signal import welch

# ── change this when real device aaye ──
USE_SYNTHETIC = True   # True = no hardware needed (testing mode)
SERIAL_PORT   = ""     # OpenBCI ke liye: "COM3" ya "/dev/ttyUSB0"
# ────────────────────────────────────────

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

BOARD_ID  = BoardIds.SYNTHETIC_BOARD if USE_SYNTHETIC else BoardIds.MUSE_2_BOARD
FS        = 256
WINDOW_SEC = 4
BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  12),
    "beta":  (12, 30),
    "gamma": (30, 40),
}

# load saved model artifacts
models_dir   = Path("future_eeg/models")
scaler       = joblib.load(models_dir / "scaler.pkl")
gmm          = joblib.load(models_dir / "gmm_model.pkl")
feature_order = joblib.load(models_dir / "feature_order.pkl")

log_path = Path("future_eeg/live_predictions.csv")

cluster_meaning = {
    0: "Mood / Anxiety pattern",
    1: "Addictive / Low-fatigue pattern",
    2: "Outlier / High-fatigue pattern",
    3: "Mixed / Baseline pattern",
}

def extract_band_powers(eeg_window, sfreq, n_channels):
    freqs, psd = welch(eeg_window, sfreq, nperseg=min(256, eeg_window.shape[1]), axis=1)
    features = {}
    for ch in range(n_channels):
        for band, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs <= hi)
            features[f"{band}_ch{ch}"] = float(np.mean(psd[ch, mask]))
    return features

def add_derived(f):
    # frontal alpha asymmetry: right(ch2/AF8) - left(ch1/AF7)
    f["alpha_asymmetry"] = f.get("alpha_ch2", 0) - f.get("alpha_ch1", 0)
    # theta/beta ratio averaged across channels
    theta = np.mean([f.get(f"theta_ch{i}", 0) for i in range(4)])
    beta  = np.mean([f.get(f"beta_ch{i}",  0) for i in range(4)])
    f["theta_beta_ratio"] = theta / beta if beta != 0 else 0.0
    return f

# Muse 2 channel → training column name mapping
# Muse:  ch0=TP9(left), ch1=AF7(frontal-left), ch2=AF8(frontal-right), ch3=TP10(right)
# Training used: F3, F4, C3, C4, Pz  (BRMH dataset: FP1, FP2, F7, F3 etc.)
# We map best-effort:
CH_MAP = {
    "ch0": "FP1",   # TP9  → closest frontal-left
    "ch1": "F3",    # AF7  → frontal left
    "ch2": "F4",    # AF8  → frontal right
    "ch3": "FP2",   # TP10 → closest frontal-right
}

def build_model_input(features):
    row = {col: 0.0 for col in feature_order}
    for band in BANDS:
        for ch_idx, train_ch in CH_MAP.items():
            live_key  = f"{band}_{ch_idx}"
            train_key = f"{band}_{train_ch}"
            if train_key in row:
                row[train_key] = features.get(live_key, 0.0)
    if "alpha_asymmetry" in row:
        row["alpha_asymmetry"]  = features.get("alpha_asymmetry", 0.0)
    if "theta_beta_ratio" in row:
        row["theta_beta_ratio"] = features.get("theta_beta_ratio", 0.0)
    return np.array([row[c] for c in feature_order]).reshape(1, -1)

# ── start stream ──
params = BrainFlowInputParams()
params.serial_port = SERIAL_PORT
board = BoardShim(BOARD_ID, params)

BoardShim.enable_dev_board_logger()
board.prepare_session()
board.start_stream()

mode = "SYNTHETIC (no hardware)" if USE_SYNTHETIC else "REAL DEVICE"
print(f"\nNeuroMosaic Live Prediction — {mode}")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        time.sleep(WINDOW_SEC)
        data = board.get_board_data()

        n_eeg = 4
        eeg   = data[:n_eeg, :]

        if eeg.shape[1] < FS * WINDOW_SEC:
            print("Waiting for enough data...")
            continue

        window   = eeg[:, -FS * WINDOW_SEC:]
        features = extract_band_powers(window, FS, n_eeg)
        features = add_derived(features)
        X_live   = build_model_input(features)
        X_scaled = scaler.transform(X_live)
        cluster  = int(gmm.predict(X_scaled)[0])
        probs    = gmm.predict_proba(X_scaled)[0]

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}]  Cluster: {cluster}  →  {cluster_meaning[cluster]}")
        print(f"         Probabilities: { {i: round(p,2) for i,p in enumerate(probs)} }")
        print(f"         Alpha Asymmetry : {features['alpha_asymmetry']:.3f}")
        print(f"         Theta/Beta Ratio: {features['theta_beta_ratio']:.3f}\n")

        log_row = {"timestamp": ts, "cluster": cluster, "meaning": cluster_meaning[cluster],
                   **{f"prob_{i}": round(p, 3) for i, p in enumerate(probs)},
                   "alpha_asymmetry": features["alpha_asymmetry"],
                   "theta_beta_ratio": features["theta_beta_ratio"]}
        pd.DataFrame([log_row]).to_csv(log_path, mode="a",
                                       header=not log_path.exists(), index=False)

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    board.stop_stream()
    board.release_session()
    print("Session closed. Predictions saved to future_eeg/live_predictions.csv")
