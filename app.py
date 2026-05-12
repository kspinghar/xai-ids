"""
app.py — Gradio demo for the hybrid XAI-IDS Space.
"""

import io
import json
import tempfile
from pathlib import Path

import gradio as gr
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import lime.lime_tabular
import xgboost as xgb
from keras.models import load_model

ART = Path(__file__).parent / "artifacts"

# ───────────────────────── load artefacts ─────────────────────────
print("Loading artefacts...")
scaler = joblib.load(ART / "scaler.joblib")
pca = joblib.load(ART / "pca.joblib")
ocsvm = joblib.load(ART / "ocsvm.joblib")
rf = joblib.load(ART / "rf.joblib")
le = joblib.load(ART / "label_encoder.joblib")
encoder = load_model(ART / "encoder.keras", compile=False)
hybrid = xgb.XGBClassifier()
hybrid.load_model(str(ART / "hybrid.json"))

feature_names = json.loads((ART / "feature_names.json").read_text())
top_features = json.loads((ART / "top_features.json").read_text())
examples = json.loads((ART / "examples.json").read_text())
example_labels = list(examples.keys())

# ───────────────────────── inference helpers ─────────────────────────
def _build_row(example_label, slider_values):
    """Take the chosen example as base, override the top features with slider values."""
    base = examples[example_label].copy()
    for fname, val in zip(top_features, slider_values):
        base[fname] = val
    return pd.DataFrame([[base[c] for c in feature_names]], columns=feature_names)

def _predict_pipeline(row_df):
    scaled = scaler.transform(row_df)
    pca_feats = pca.transform(scaled)
    encoded = encoder.predict(pca_feats, verbose=0)
    svm_pred = 0 if ocsvm.predict(encoded)[0] == 1 else 1
    rf_pred = rf.predict(scaled)[0]
    rf_proba = rf.predict_proba(scaled)[0]
    hybrid_input = np.array([[svm_pred, rf_pred]])
    hybrid_pred = int(hybrid.predict(hybrid_input)[0])
    hybrid_proba = float(hybrid.predict_proba(hybrid_input)[0][hybrid_pred])
    rf_label = le.inverse_transform([rf_pred])[0]
    return {
        "svm_pred": svm_pred,
        "rf_pred": rf_pred,
        "rf_label": rf_label,
        "hybrid_pred": hybrid_pred,
        "hybrid_proba": hybrid_proba,
        "scaled": scaled,
        "rf_proba": rf_proba,
    }

def _shap_plot(svm_pred, rf_pred):
    """Run SHAP TreeExplainer on the hybrid XGBoost; render waterfall as PNG file."""
    explainer = shap.TreeExplainer(hybrid)
    sv = explainer(np.array([[svm_pred, rf_pred]]))
    fig = plt.figure(figsize=(6, 3))
    shap.plots.waterfall(sv[0], show=False, max_display=2)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.tight_layout()
    plt.savefig(tmp.name, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

def _lime_html(scaled_row):
    """LIME explanation on the Random Forest, returned as HTML."""
    ref = np.array([
        [examples[k][f] for f in feature_names] for k in example_labels
    ])
    ref_scaled = scaler.transform(ref)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        ref_scaled,
        feature_names=feature_names,
        class_names=[str(c) for c in le.classes_],
        discretize_continuous=True,
    )
    exp = explainer.explain_instance(
        scaled_row.flatten(),
        rf.predict_proba,
        num_features=8,
    )
    # Render LIME's feature/weight pairs as a clean HTML table styled for dark mode.
    # LIME's own as_html() relies on inline JS that doesn't run inside Gradio's iframe.
    pairs = exp.as_list()
    rows_html = ""
    max_abs = max((abs(w) for _, w in pairs), default=1.0)
    for feat, weight in pairs:
        color = "#ef4444" if weight > 0 else "#3b82f6"  # red for attack-pushing, blue for benign-pushing
        bar_pct = (abs(weight) / max_abs) * 100 if max_abs > 0 else 0
        rows_html += (
            f"<tr>"
            f"<td style='padding:6px 10px;color:#e8e8f0;font-family:monospace;font-size:0.8rem'>{feat}</td>"
            f"<td style='padding:6px 10px;text-align:right;color:{color};font-family:monospace;font-size:0.8rem'>{weight:+.4f}</td>"
            f"<td style='padding:6px 10px;width:200px'>"
            f"<div style='background:{color};height:10px;width:{bar_pct:.1f}%;border-radius:2px'></div>"
            f"</td>"
            f"</tr>"
        )
    return (
        f"<div style='background:#12121e;border:1px solid #1e1e2e;border-radius:6px;padding:12px'>"
        f"<div style='color:#a0a0b8;font-size:0.75rem;margin-bottom:8px;font-family:monospace;text-transform:uppercase;letter-spacing:0.1em'>"
        f"LIME local explanation — top {len(pairs)} features"
        f"</div>"
        f"<table style='width:100%;border-collapse:collapse'>{rows_html}</table>"
        f"<div style='color:#6b7280;font-size:0.7rem;margin-top:8px'>"
        f"Red bars push toward attack, blue bars push toward benign. Length reflects relative weight."
        f"</div>"
        f"</div>"
    )

# ───────────────────────── main predict function for gradio ─────────────────────────
def predict(example_label, *slider_values):
    row_df = _build_row(example_label, slider_values)
    result = _predict_pipeline(row_df)
    verdict = "ATTACK" if result["hybrid_pred"] == 1 else "BENIGN"
    summary = (
        f"## Verdict: {verdict}\n\n"
        f"**Hybrid confidence:** {result['hybrid_proba']*100:.1f}%\n\n"
        f"**Random Forest fine-grained label:** {result['rf_label']}\n\n"
        f"**Anomaly branch (OCSVM):** {'anomaly' if result['svm_pred'] == 1 else 'normal'}"
    )
    shap_path = _shap_plot(result["svm_pred"], result["rf_pred"])
    lime_html = _lime_html(result["scaled"])
    return summary, shap_path, lime_html

# ───────────────────────── slider construction ─────────────────────────
def _slider_for(fname):
    """Build a slider with reasonable bounds drawn from the example rows."""
    values = [examples[k][fname] for k in example_labels]
    lo, hi = float(min(values)), float(max(values))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    step = max((hi - lo) / 100, 1e-3)
    return gr.Slider(minimum=lo, maximum=hi, step=step, value=values[0], label=fname)

# ───────────────────────── gradio UI ─────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="XAI Intrusion Detection") as demo:
    gr.Markdown(
        "# XAI Intrusion Detection System\n"
        "Pick an example flow, optionally tweak the top features, hit **Predict**. "
        "The hybrid model decides; SHAP and LIME explain why.\n\n"
        "*First load may take ~30s while the Space wakes up.*"
    )
    with gr.Row():
        with gr.Column(scale=1):
            example_dd = gr.Dropdown(
                choices=example_labels, value=example_labels[0],
                label="Example flow",
            )
            sliders = [_slider_for(f) for f in top_features]
            predict_btn = gr.Button("Predict", variant="primary")
        with gr.Column(scale=2):
            output_md = gr.Markdown()
            shap_img = gr.Image(label="SHAP waterfall (hybrid head)", type="filepath")
            lime_out = gr.HTML(label="LIME local explanation")

    predict_btn.click(
        predict,
        inputs=[example_dd] + sliders,
        outputs=[output_md, shap_img, lime_out],
    )

if __name__ == "__main__":
    demo.launch(share=True, server_port=7860, show_api=False)
