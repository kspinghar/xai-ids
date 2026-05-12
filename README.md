---
title: XAI Intrusion Detection
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# XAI Intrusion Detection System

A scaled-down, interactive demonstrator of the hybrid intrusion detection system from my bachelor thesis. Combines an Autoencoder + One-Class SVM anomaly detector with a Random Forest signature detector, with an XGBoost meta-classifier stacking the two. Predictions are accompanied by SHAP global feature importance and LIME local explanations.

**Note:** This Space runs on a 50,000-row stratified sample of CIC-IDS-2017, not the full 2.8M-row dataset used in the thesis. The full thesis results — including hyperparameter searches and the 15-class breakdown — are in the PDF on my portfolio.

## What this demo does

1. Pick one of five pre-baked example flows (Benign, DoS Hulk, PortScan, FTP-Patator, DDoS) from a dropdown.
2. Optionally tweak the 8 most-important features using sliders.
3. Hit **Predict** to see the hybrid model's classification, the SHAP waterfall plot, and a LIME local explanation.

## Architecture

- Anomaly branch: `StandardScaler` → `PCA(30)` → `RandomOverSampler` → Autoencoder (encoding_dim=128) → encoded features → One-Class SVM (`nu=0.1`, RBF kernel).
- Signature branch: `StandardScaler` → `RandomOverSampler` → `RandomForestClassifier(n_estimators=100, max_depth=20)`.
- Meta-classifier: `XGBClassifier` trained on the stacked anomaly + signature predictions.
- Explainers: SHAP `TreeExplainer` on the hybrid, LIME `LimeTabularExplainer` on the Random Forest.

## Training the artefacts

The artefacts shipped with this Space were produced by `train_and_save.py` from the dataset in the linked GitHub repo. Re-running is not required to use the demo.
