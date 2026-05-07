# NeuroMosaic — Future EEG Live Prediction System

Ye folder tab use karna jab real EEG machine aaye.
Abhi Kaggle data wali pipeline bilkul safe hai — kuch touch nahi hua.

---

## Folder Structure

```
future_eeg/
├── save_model.py          ← Step 1: Run this ONCE to save trained model
├── live_predict.py        ← Step 2: Run this when EEG machine connected
├── models/
│   ├── scaler.pkl         ← StandardScaler (saved after save_model.py)
│   ├── gmm_model.pkl      ← Trained GMM k=4 model
│   └── feature_order.pkl  ← Exact feature column order model expects
└── live_predictions.csv   ← Auto-created, logs every prediction
```

---

## Step 1 — Model Save Karo (Abhi Karo, Ek Baar)

```
python future_eeg/save_model.py
```

Ye current Kaggle-trained model ko `future_eeg/models/` mein save kar dega.

---

## Step 2 — Jab EEG Machine Aaye

### Option A: Bina Machine Ke Test (Synthetic Mode)
`live_predict.py` mein ye line already set hai:
```python
USE_SYNTHETIC = True
```
Bas chalao:
```
pip install brainflow
python future_eeg/live_predict.py
```
Fake EEG data generate hoga aur cluster predict hoga — real machine jaisa hi flow.

### Option B: Muse 2 Headband (Real Device)
1. Muse 2 ko Bluetooth se connect karo
2. `live_predict.py` mein change karo:
```python
USE_SYNTHETIC = False
```
3. Chalao:
```
python future_eeg/live_predict.py
```

### Option C: OpenBCI
1. `live_predict.py` mein:
```python
USE_SYNTHETIC = False
SERIAL_PORT   = "COM3"   # apna port dekho Device Manager mein
BOARD_ID      = BoardIds.CYTON_BOARD
```

---

## Cluster Meanings (Current Model)

| Cluster | Pattern |
|---------|---------|
| 0 | Mood / Anxiety dominant |
| 1 | Addictive / Low-fatigue |
| 2 | Outlier / High-fatigue |
| 3 | Mixed / Baseline |

---

## Important Notes

- Kaggle pipeline (`load_brmh_data.py`, `force_four_clusters.py` etc.) bilkul safe hai
- Ye folder completely independent hai
- `save_model.py` ek baar chalane ke baad `models/` folder ready ho jaata hai
- Jab naya data aaye (apna ya kisi aur ka), `live_predict.py` seedha predict karega
- Har prediction `live_predictions.csv` mein save hoti hai — apna personal database

---

## Recommended EEG Devices (Budget ke hisaab se)

| Device | Price | Channels | Best For |
|--------|-------|----------|----------|
| Muse 2 | ~₹20,000 | 4 | Beginners, meditation tracking |
| Muse S | ~�25,000 | 4 | Sleep tracking bhi |
| OpenBCI Cyton | ~₹35,000 | 8 | Research grade |
| OpenBCI Ganglion | ~₹20,000 | 4 | Budget research |
