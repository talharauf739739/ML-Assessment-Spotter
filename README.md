# Spotter Freight Rate ML Prediction Challenge

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Complete](https://img.shields.io/badge/Status-Complete-success.svg)]()

This repository contains the end-to-end Machine Learning pipeline developed for the **Spotter Freight Rate Prediction Assessment**. The objective is to predict spot freight load rates (`posted_rate`) across continental US freight lanes, ensuring high quote accuracy, robustness against seasonal volatility, and compliance with Spotter's official evaluation requirements.

---

## 1. Repository Architecture

```
data/
├── train_test.csv                    # 48,000 labeled development loads (Jan - Oct 2025)
├── validation.csv                    # 12,000 unlabeled out-of-time loads (Nov - Dec 2025)
├── december_chart_inputs.csv         # 31 daily scenario loads with filled predictions
└── validation_predictions_template.csv
src/
├── __init__.py
├── config.py                         # Single source of truth for paths, parameters & schemas
├── eda.py                            # Data audit & exploratory visualizations
├── features.py                       # Modular spatial, temporal, work & market feature pipeline
├── models.py                         # Baselines, XGBoost, LightGBM, CatBoost & Ensembles
├── train.py                          # Anti-leakage out-of-time validation & model benchmarking
└── inference.py                      # Production batch inference & score.py execution
scorer_results/
├── candidate_december.png            # Verified December 2025 rate chart produced by score.py
├── eda_distributions.png             # Distance and rate per mile histograms
├── eda_equipment_rates.png           # Equipment rate per mile distribution
├── eda_temporal_trend.png            # 2025 daily freight rate volatility
├── model_benchmark_comparison.png    # MAE and R² comparison across candidates
└── validation_metrics.json           # Serialized out-of-time benchmark metrics
score.py                              # Official Spotter evaluation and charting script
validation_predictions.csv            # 12,000 final predictions (load_id, predicted_rate)
METHODOLOGY.md                        # Formal engineering methodology specification
REPORT.md                             # Comprehensive technical report
README.md                             # Reproducibility, installation & run instructions
requirements.txt                      # Project dependencies
```

---

## 2. Quickstart & Installation

### Prerequisites
* Python 3.11 or higher
* Git

```bash
# 1. Clone the repository
git clone https://github.com/talharauf739739/ML-Assessment-Spotter.git
cd ML-Assessment-Spotter

# 2. Checkout the assessment branch
git checkout assessment

# 3. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .venv\Scripts\activate        # On Windows (PowerShell/CMD)

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 3. Running the Pipeline

You can run each stage independently or execute the complete workflow:

### Step 1: Run Exploratory Data Analysis & Audit
```bash
python src/eda.py
```
* Audits missing values and distributions across datasets.
* Generates exploratory plots in `scorer_results/` (`eda_distributions.png`, `eda_equipment_rates.png`, `eda_temporal_trend.png`).

### Step 2: Run Out-of-Time Training & Model Benchmarking
```bash
python src/train.py
```
* Implements a strict **Out-of-Time chronological split** (Train: Jan 1 – Aug 31, 2025; Validation: Sep 1 – Oct 31, 2025) to prevent temporal data leakage.
* Benchmarks Naive Equipment baseline, Ridge, XGBoost, LightGBM, CatBoost, and Blended Ensembles.
* Saves `scorer_results/validation_metrics.json` and `scorer_results/model_benchmark_comparison.png`.

### Step 3: Run Full Inference & Score Verification
```bash
python src/inference.py
```
* Retrains the best ensemble pipeline on the full historical dataset (Jan–Oct 2025).
* Generates `validation_predictions.csv` (12,000 loads strictly formatted as `load_id,predicted_rate`).
* Completes `data/december_chart_inputs.csv` with daily rate forecasts for the fixed Lexington $\to$ Fort Wayne lane.
* Automatically triggers `score.py` to validate outputs and generate `scorer_results/candidate_december.png`.

---

## 4. Validating Directly with Spotter Scorer

To run Spotter’s official verification script directly:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

**Expected Output**:
```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

---

## 5. Summary of Out-of-Time Benchmark Results

Evaluated on the 2-month out-of-time holdout (Sep 1 – Oct 31, 2025; 9,523 loads):

| Model | Target Formulation | MAE ($) | RMSE ($) | WAPE (%) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Naive Baseline** | Equipment Mean RPM | $270.71 | $696.74 | 11.32% | 0.7915 |
| **Ridge Regression** | Rate Per Mile (RPM) | $186.71 | $673.28 | 7.80% | 0.8054 |
| **XGBoost Regressor** | Rate Per Mile (RPM) | $152.72 | $659.52 | 6.38% | 0.8132 |
| **LightGBM Regressor** | Rate Per Mile (RPM) | $151.80 | $644.53 | 6.34% | 0.8216 |
| **CatBoost Regressor** | Rate Per Mile (RPM) | **$117.42** | **$635.13** | **4.91%** | **0.8268** |
| **Blended Ensemble** | Weighted Blend | **$119.97** | **$635.78** | **5.01%** | **0.8264** |

---

## 6. Deliverables Summary

* **GitHub Repository**: Accessible at [`talharauf739739/ML-Assessment-Spotter`](https://github.com/talharauf739739/ML-Assessment-Spotter) on branch `assessment`.
* **Predictions File**: [`validation_predictions.csv`](validation_predictions.csv) (12,000 rows).
* **December Scorer Chart**: [`scorer_results/candidate_december.png`](scorer_results/candidate_december.png).
* **Comprehensive Technical Report**: [`REPORT.md`](REPORT.md) (ready for PDF/DOCX conversion).
* **Detailed Methodology**: [`METHODOLOGY.md`](METHODOLOGY.md).
