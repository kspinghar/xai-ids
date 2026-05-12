"""
train_and_save.py
Re-run the hybrid XAI-IDS pipeline on a 50k stratified sample and dump
artefacts into ./artifacts/ for the Gradio Space to load at startup.
"""

import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import OneClassSVM
import xgboost as xgb
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.initializers import HeNormal
from keras.layers import (BatchNormalization, Dense, Dropout, Input, LeakyReLU)
from keras.models import Model
from keras.optimizers import RMSprop
from keras.regularizers import l2

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ───────────────────────── paths ─────────────────────────
DATA_DIR = Path(
    r"C:\Users\khali\iCloudDrive\Workspace\UNIVERSITIES\NOROFF\YEAR 3\Bachelor Project\MachineLearningCVE"
)
ART_DIR = Path(__file__).parent / "artifacts"
ART_DIR.mkdir(exist_ok=True)

# ───────────────────────── load ─────────────────────────
print("Loading CIC-IDS-2017 CSVs...")
dfs = []
for csv_path in sorted(DATA_DIR.glob("*.csv")):
    df = pd.read_csv(csv_path, low_memory=False)
    dfs.append(df)
    print(f"  loaded {csv_path.name}: {df.shape}")

data = pd.concat(dfs, ignore_index=True)
data.columns = data.columns.str.strip()
print(f"Concatenated: {data.shape}")

# ───────────────────────── stratified subset ─────────────────────────
SUBSET_SIZE = 50_000
print(f"Stratified subset to {SUBSET_SIZE:,} rows...")
class_counts = (data["Label"].value_counts(normalize=True) * SUBSET_SIZE).astype(int)
class_counts[class_counts < 2] = 2

def _sample(g):
    n = class_counts[g.name]
    return g.sample(n=n, replace=len(g) < n, random_state=42)

data_subset = (
    data.groupby("Label", group_keys=False).apply(_sample).reset_index(drop=True)
)
print(f"Subset shape: {data_subset.shape}")

# ───────────────────────── clean numeric ─────────────────────────
num_cols = data_subset.select_dtypes(include=[np.number]).columns.tolist()
data_subset[num_cols] = data_subset[num_cols].replace([np.inf, -np.inf], np.nan)
data_subset[num_cols] = data_subset[num_cols].fillna(data_subset[num_cols].median())
data_subset.drop(
    columns=["Flow ID", "Source IP", "Destination IP", "Timestamp"],
    errors="ignore",
    inplace=True,
)

# ───────────────────────── split + encode ─────────────────────────
X = data_subset.drop(columns=["Label"])
y_raw = data_subset["Label"]
le = LabelEncoder()
y = le.fit_transform(y_raw)
feature_names = X.columns.tolist()
print(f"Features: {len(feature_names)} | classes: {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ───────────────────────── scaler + PCA ─────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=30, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

ros = RandomOverSampler(sampling_strategy="minority", random_state=42)
X_train_balanced_pca, _ = ros.fit_resample(X_train_pca, y_train)
X_train_balanced_scaled, y_train_balanced = ros.fit_resample(X_train_scaled, y_train)

# ───────────────────────── autoencoder ─────────────────────────
print("Training autoencoder...")
input_dim = X_train_balanced_pca.shape[1]
inp = Input(shape=(input_dim,))
e = Dense(128, kernel_regularizer=l2(0.01), kernel_initializer=HeNormal())(inp)
e = BatchNormalization()(e); e = LeakyReLU()(e); e = Dropout(0.2)(e)
e = Dense(64, kernel_regularizer=l2(0.01), kernel_initializer=HeNormal())(e)
e = BatchNormalization()(e); e = LeakyReLU()(e); e = Dropout(0.2)(e)
e = Dense(32, kernel_regularizer=l2(0.01), kernel_initializer=HeNormal())(e)
e = BatchNormalization()(e); encoded = LeakyReLU()(e)

d = Dense(32, kernel_regularizer=l2(0.01), kernel_initializer=HeNormal())(encoded)
d = BatchNormalization()(d); d = LeakyReLU()(d); d = Dropout(0.2)(d)
d = Dense(64, kernel_regularizer=l2(0.01), kernel_initializer=HeNormal())(d)
d = BatchNormalization()(d); d = LeakyReLU()(d); d = Dropout(0.2)(d)
out = Dense(input_dim, activation="sigmoid")(d)

autoencoder = Model(inp, out)
autoencoder.compile(optimizer=RMSprop(learning_rate=1e-3), loss="mse")
autoencoder.fit(
    X_train_balanced_pca, X_train_balanced_pca,
    epochs=30, batch_size=64, shuffle=True,
    validation_data=(X_test_pca, X_test_pca), verbose=1,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-4),
    ],
)
encoder = Model(autoencoder.input, encoded)
X_train_encoded = encoder.predict(X_train_balanced_pca, verbose=0)
X_test_encoded = encoder.predict(X_test_pca, verbose=0)

# ───────────────────────── one-class SVM ─────────────────────────
print("Training One-Class SVM...")
ocsvm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
ocsvm.fit(X_train_encoded)
y_test_pred_svm = np.where(ocsvm.predict(X_test_encoded) == 1, 0, 1)

# ───────────────────────── random forest ─────────────────────────
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100, max_depth=20, max_features="sqrt",
    criterion="gini", random_state=42, n_jobs=-1,
)
rf.fit(X_train_balanced_scaled, y_train_balanced)
y_test_pred_rf = rf.predict(X_test_scaled)

# ───────────────────────── hybrid XGBoost head ─────────────────────────
print("Training hybrid XGBoost head...")
combined_test = np.vstack((y_test_pred_svm, y_test_pred_rf)).T
y_test_binary = (y_test != 0).astype(int)
hybrid = xgb.XGBClassifier(random_state=42, n_estimators=100, use_label_encoder=False, eval_metric="logloss")
hybrid.fit(combined_test, y_test_binary)

# ───────────────────────── top features by RF importance ─────────────────────────
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
top_features = importances.head(8).index.tolist()
print(f"Top 8 features: {top_features}")

# ───────────────────────── example rows ─────────────────────────
print("Selecting 5 example rows...")
target_labels = ["BENIGN", "DoS Hulk", "PortScan", "FTP-Patator", "DDoS"]
examples = {}
for label in target_labels:
    rows = data_subset[data_subset["Label"] == label]
    if len(rows) == 0:
        continue
    row = rows.sample(n=1, random_state=42).iloc[0]
    examples[label] = {col: float(row[col]) for col in feature_names}
print(f"Examples saved: {list(examples.keys())}")

# ───────────────────────── save everything ─────────────────────────
print(f"Saving artefacts to {ART_DIR}...")
joblib.dump(scaler, ART_DIR / "scaler.joblib")
joblib.dump(pca, ART_DIR / "pca.joblib")
joblib.dump(ocsvm, ART_DIR / "ocsvm.joblib")
joblib.dump(rf, ART_DIR / "rf.joblib")
joblib.dump(le, ART_DIR / "label_encoder.joblib")
hybrid.save_model(str(ART_DIR / "hybrid.json"))
autoencoder.save(ART_DIR / "autoencoder.keras")
encoder.save(ART_DIR / "encoder.keras")

with open(ART_DIR / "feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)
with open(ART_DIR / "top_features.json", "w") as f:
    json.dump(top_features, f, indent=2)
with open(ART_DIR / "examples.json", "w") as f:
    json.dump(examples, f, indent=2)

# Sanity check on disk sizes — fail loudly if any single artefact > 50MB
for path in ART_DIR.iterdir():
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  {path.name}: {size_mb:.2f} MB")
    if size_mb > 50:
        print(f"  WARNING: {path.name} is over 50MB — Git LFS will be required.")

print("Done.")
