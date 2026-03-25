# Adaptive STRIDE-based Multi-Label Intrusion Detection
### AI-Augmented Behavioral Modeling for Cybersecurity Threat Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Research%20Active-orange)

---

## Overview

Most intrusion detection systems answer one question:
**"Is this traffic an attack?"**

This project answers a better one:
**"Which security properties are being violated — and how severely?"**

We reformulate network intrusion detection as structured
multi-label STRIDE vector prediction:

f(x) → [Spoofing, Tampering, Repudiation,
         Info Disclosure, DoS, Elevation of Privilege]

Instead of one label per flow, every network connection
receives a 6-dimensional threat profile simultaneously.

---

## Key Results

| Metric | UNSW-NB15 | CIC-IDS-2018 |
|--------|-----------|--------------|
| Macro AUROC | 0.9556 ± 0.0006 | 0.9893 |
| Macro F1 | 0.6869 | 0.7896 |
| Hamming Loss | 0.0705 | 0.0358 |
| McNemar p-value | < 0.0001 | < 0.0001 |

✅ Statistically significant improvement over
   single-label baseline across all 6 STRIDE dimensions

---

## Architecture

UNSW-NB15 / CIC-IDS-2018
        ↓
STRIDE Label Mapping (6-dim vectors)
        ↓
Behavioural Feature Engineering (8 signals, 202 features)
        ↓
    ┌───────────────────────┐
    │   XGBoost (static)    │
    │   Bi-LSTM (temporal)  │
    └───────────┬───────────┘
                ↓
        Hybrid Fusion
    (Late Avg / Attn / Meta-Stack)
                ↓
    Threshold Optimisation
                ↓
    SHAP Semantic Validation
                ↓
    [S, T, R, I, D, E] Threat Profile

---

## Pipeline Steps

1. STRIDE multi-label mapping (10 attack categories → 6-dim)
2. 8 STRIDE-aligned behavioural signals engineered
3. XGBoost with 6 parallel binary classifiers
4. Bidirectional LSTM with attention (10-flow window)
5. Three hybrid fusion strategies
6. Per-label threshold optimisation
7. 5-fold stratified cross-validation
8. SHAP explainability analysis
9. McNemar statistical significance test
10. CIC-IDS-2018 cross-dataset validation

---

## SHAP Findings

| Feature | Top STRIDE Dim | SHAP Value |
|---------|---------------|------------|
| sttl | ALL 6 dims | 0.876–1.40 |
| service_dns | DoS | 1.560 |
| feat_asymmetry_ratio ★ | Repudiation | 0.847 |
| proto_udp | Repudiation | 1.165 |

★ Engineered behavioural signal

---

## Datasets

This project uses two publicly available datasets:

- **UNSW-NB15** — [Download from Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
- **CIC-IDS-2018** — [Download from Kaggle](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv)

Place CSV files in the `data/` directory before running.

---

## Installation

git clone https://github.com/YOUR_USERNAME/STRIDE-MultiLabel-IDS
cd STRIDE-MultiLabel-IDS
pip install -r requirements.txt

---

## Requirements

xgboost>=2.0
tensorflow>=2.13
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
shap>=0.42
matplotlib>=3.7
statsmodels>=0.14
imbalanced-learn>=0.11

---

## Results & Figures

All generated figures are in the `results/` folder:
- ROC curves per STRIDE dimension
- Confusion matrices (threshold-optimised)
- Precision-Recall curves (both datasets)
- SHAP feature importance table
- Training time comparison

---

## Status

- Paper in preparation — target: IEEE Access
- Thesis B & C: Complete
- Cross-dataset validation: Complete
- Statistical significance testing: Complete

---

## Citation

If you use this work, please cite:

@article{zulkifli2024stride,
  title   = {Adaptive STRIDE-based Multi-Label Intrusion
             Detection using AI-Augmented Behavioral Modeling},
  author  = {Zulkifli, Muhammad Hakiem Bin},
  journal = {Under Review — IEEE Access},
  year    = {2024}
}

---

## Author

NETIK KUMAR MAHESHWARI 
Thesis — AI-Augmented Behavioral Modeling for Cybersecurity Threat Detection
