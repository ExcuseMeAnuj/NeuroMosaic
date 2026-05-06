# NeuroMosaic — Complete Setup & Run Instructions

Mental health EEG clustering project using Kaggle + BRMH datasets.

---

## Project Structure

```
NeuroMosaic/
├── data/
│   ├── raw/                    # Raw EEG datasets (download separately)
│   └── processed/              # Processed combined data
├── results/                    # Clustering results & visualizations
├── future_eeg/                 # Live EEG prediction system (future use)
├── *.py                        # Main pipeline scripts
└── SETUP_INSTRUCTIONS.md       # This file
```

---

##  Quick Start (5 Steps)

### Step 1: Clone Repository
```bash
git clone https://github.com/excusemeanuj/NeuroMosaic.git
cd NeuroMosaic
```
[Open folder In VS Code ]
### Step 2: Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn plotly umap-learn mne
```
[ Open terminal in VS CODE By ctrl + ` Run these for Different files 
python generate_neuromosaic_data.py
python load_brmh_data.py
python merge_neuromosaic_data.py
python find_optimal_k.py
python force_four_clusters.py
python report.py
python view_report.py ] 

### Step 3: Download Raw Data
Place these files in `data/raw/`:
- `EEG.machinelearing_data_BRMH.csv` (BRMH dataset)
- `clinical_scores.csv` (Clinical scores)
- `eeg_features.csv` (Kaggle EEG features)
- `ds003478/` folder (Optional: BIDS format EEG data)

### Step 4: Run Pipeline (In Order)
```bash
# Generate synthetic Kaggle-like data
python generate_neuromosaic_data.py

# Load BRMH dataset
python load_brmh_data.py

# Merge both datasets
python merge_neuromosaic_data.py

# Find optimal clusters
python find_optimal_k.py

# Force 4 clusters (final model)
python force_four_clusters.py

# Generate interactive report
python report.py

# View report in browser
python view_report.py
```

---

##  What Each Script Does

| Script | Purpose | Output |
|--------|---------|--------|
| `generate_neuromosaic_data.py` | Creates synthetic Kaggle-style EEG features | `data/processed/kaggle_synthetic.csv` |
| `load_brmh_data.py` | Loads & processes BRMH dataset | `data/processed/brmh_processed.csv` |
| `merge_neuromosaic_data.py` | Combines both datasets | `data/processed/combined_data.csv` |
| `find_optimal_k.py` | Tests k=2 to k=10 clusters | `results/figures/optimal_k.png` |
| `force_four_clusters.py` | Trains GMM with k=4 | `results/clustered_subjects_4.csv` |
| `report.py` | Generates interactive HTML report | `results/report.html` |
| `view_report.py` | Opens report in browser | - |
| `explore_neuromosaic.py` | Quick data exploration | Console output |

---

## Understanding the 4 Clusters

| Cluster | Name | Characteristics |
|---------|------|-----------------|
| 0 | **Anxious Distress** | High anxiety, moderate depression, activated EEG |
| 1 | **Burnout/Exhaustion** | Mixed depression/stress, elevated theta/beta |
| 2 | **Healthy/Resilient** | Low clinical burden, balanced EEG |
| 3 | **Melancholic Depression** | Strong depression, low anxiety, negative alpha asymmetry |

---

## 🔮 Future: Live EEG Prediction

### When You Get a Real EEG Device

#### Step 1: Save Current Model (Run Once)
```bash
python future_eeg/save_model.py
```
This saves trained model to `future_eeg/models/`

#### Step 2: Test with Synthetic Data
```bash
pip install brainflow
python future_eeg/live_predict.py
```
Generates fake EEG data and predicts cluster (for testing)

#### Step 3: Connect Real Device
Edit `future_eeg/live_predict.py`:
```python
USE_SYNTHETIC = False  # Change to False
SERIAL_PORT = "COM3"   # Your device port (check Device Manager)
```

Then run:
```bash
python future_eeg/live_predict.py
```

**Supported Devices:**
- Muse 2 / Muse S (Bluetooth)
- OpenBCI Cyton / Ganglion (Serial)
- Any BrainFlow-compatible device

---

## 📦 Required Python Packages

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
umap-learn>=0.5.0
mne>=1.0.0
brainflow>=5.0.0  # Only for live prediction
```

Save as `requirements.txt` and install:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Troubleshooting

### Issue: "File not found" error
**Solution:** Make sure raw data files are in `data/raw/` folder

### Issue: "Module not found" error
**Solution:** Install missing package:
```bash
pip install <package_name>
```

### Issue: Report not opening
**Solution:** Manually open `results/report.html` in browser

### Issue: Slow processing
**Solution:** Reduce CPU threads in `report.py`:
```python
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")  # Change 4 to 2
```

---

## Expected Results

After running full pipeline:
- `results/clustered_subjects_4.csv` — 500+ subjects with cluster labels
- `results/report.html` — Interactive visualizations
- `results/figures/` — Static plots (optimal_k, clusters, etc.)

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

##  License

This project is open source and available under the MIT License.

---

## Contact

For questions or issues, open a GitHub issue or contact the maintainer.

---

## Acknowledgments

- BRMH Dataset: [Source]
- Kaggle EEG Dataset: [Source]
- MNE-Python for EEG processing
- Scikit-learn for clustering algorithms
