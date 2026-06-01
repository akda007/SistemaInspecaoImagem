import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import os
import glob
import time
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc
)

# ── Optional heavy imports ───────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    CNN_AVAILABLE = True
except Exception:
    CNN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FruitVision AI",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,700;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }

h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 700; }

.block-container { padding: 2rem 2.5rem; max-width: 1400px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #2a2d3a;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Cards */
.card {
    background: #1a1d2e;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 4px solid #f97316;
}

/* Metric badges */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}
.badge-fresh  { background: #14532d; color: #4ade80; border: 1px solid #16a34a; }
.badge-rotten { background: #450a0a; color: #f87171; border: 1px solid #dc2626; }
.badge-neutral{ background: #1e293b; color: #94a3b8; border: 1px solid #334155; }

/* Section headers */
.section-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.5rem;
}

/* Metric value */
.big-metric {
    font-family: 'Fraunces', serif;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
}

/* Prediction bar */
.pred-bar-wrap { background: #1e293b; border-radius: 6px; height: 8px; margin: 6px 0; }
.pred-bar { height: 8px; border-radius: 6px; }

/* Status dot */
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.dot-green  { background: #4ade80; }
.dot-red    { background: #f87171; }
.dot-yellow { background: #fbbf24; }

/* Table styling */
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.styled-table th {
    background: #1e293b; color: #94a3b8;
    padding: 0.5rem 0.75rem; text-align: left;
    font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase;
    border-bottom: 1px solid #334155;
}
.styled-table td {
    padding: 0.5rem 0.75rem; border-bottom: 1px solid #1e293b; color: #e2e8f0;
}
.styled-table tr:hover td { background: #1e293b; }
.best-row td { color: #f97316 !important; font-weight: 600; }

</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HELPERS — PREPROCESSING & SEGMENTATION
# ════════════════════════════════════════════════════════════════════════════

def normalize_lighting(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def keep_largest(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == biggest).astype(np.uint8) * 255


def segment_otsu(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (7, 7), 0)
    tv, m_inv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, m_bin  = cv2.threshold(g, tv, 255, cv2.THRESH_BINARY)
    h, w = g.shape; mg = 10
    mask = m_inv if m_inv[mg:h-mg, mg:w-mg].mean() >= m_bin[mg:h-mg, mg:w-mg].mean() else m_bin
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return keep_largest(mask).astype(np.uint8)


def segment_grabcut(img):
    h, w = img.shape[:2]
    MIN_M = max(5, int(min(h, w) * 0.05))
    seed  = segment_otsu(img)
    cnts, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        x1 = max(MIN_M, x - 5); y1 = max(MIN_M, y - 5)
        x2 = min(w - MIN_M, x + bw + 5); y2 = min(h - MIN_M, y + bh + 5)
        rect = (x1, y1, x2 - x1, y2 - y1)
    else:
        rect = (MIN_M, MIN_M, w - 2*MIN_M, h - 2*MIN_M)
    rx, ry, rw, rh = rect
    if rx < 1 or ry < 1 or (rx+rw) >= w or (ry+rh) >= h:
        rect = (MIN_M, MIN_M, w - 2*MIN_M, h - 2*MIN_M)
    gc_mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, gc_mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        gc_mask = np.where((gc_mask == cv2.GC_FGD)|(gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        gc_mask = keep_largest(gc_mask)
        gc_mask = cv2.morphologyEx(gc_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    except cv2.error:
        gc_mask = seed
    return gc_mask.astype(np.uint8)


def extract_features(img_rgb, mask):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.max() == 1: mask = mask * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c    = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    per  = cv2.arcLength(c, True)
    f = {}
    f["area"]        = area
    f["perimeter"]   = per
    f["circularity"] = (4*np.pi*area)/(per**2+1e-6)
    x, y, w, h = cv2.boundingRect(c)
    f["aspect_ratio"]  = w/(h+1e-6)
    f["extent"]        = area/(w*h+1e-6)
    hull_area          = cv2.contourArea(cv2.convexHull(c))
    f["solidity"]      = area/(hull_area+1e-6)
    if len(c) >= 5:
        (_, _), (ma, MA), _ = cv2.fitEllipse(c)
        f["eccentricity"] = np.sqrt(1-(min(ma,MA)/max(ma,MA))**2) if max(ma,MA) > 0 else 0
    else:
        f["eccentricity"] = 0
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    for i, v in enumerate(hu):
        f[f"hu_moment_{i}"] = -np.sign(v)*np.log10(abs(v)+1e-10) if v != 0 else 0
    mean_rgb = cv2.mean(img_rgb, mask=mask)
    f["mean_R"] = mean_rgb[0]; f["mean_G"] = mean_rgb[1]; f["mean_B"] = mean_rgb[2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)
    f["mean_H"] = mean_hsv[0]; f["mean_S"] = mean_hsv[1]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    pixels = gray_masked[mask > 0]
    if len(pixels) > 100:
        side    = int(np.sqrt(len(pixels)))
        tex_img = pixels[:side*side].reshape(side, side)
        glcm    = graycomatrix(tex_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        f["glcm_contrast"]    = graycoprops(glcm, "contrast")[0, 0]
        f["glcm_correlation"] = graycoprops(glcm, "correlation")[0, 0]
        f["glcm_energy"]      = graycoprops(glcm, "energy")[0, 0]
        f["glcm_homogeneity"] = graycoprops(glcm, "homogeneity")[0, 0]
    else:
        f["glcm_contrast"] = f["glcm_correlation"] = f["glcm_energy"] = f["glcm_homogeneity"] = 0
    return f


def classify_rf(feats, clf, scaler, cols):
    df  = pd.DataFrame([feats]).reindex(columns=cols, fill_value=0)
    X   = scaler.transform(df)
    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X)[0]
    classes = clf.classes_
    prob_dict = dict(zip(classes, prob))
    return pred, prob_dict


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_all_models():
    models = {}
    base = "outputs/models"
    pairs = [
        ("rf_otsu",     "RF",  "Otsu"),
        ("rf_grabcut",  "RF",  "GrabCut"),
        ("svm_otsu",    "SVM", "Otsu"),
        ("svm_grabcut", "SVM", "GrabCut"),
    ]
    for key, clf_type, seg in pairs:
        clf_path = f"{base}/{key}.pkl"
        scl_path = f"{base}/scaler_{'otsu' if 'otsu' in key else 'grabcut'}.pkl"
        fts_path = f"{base}/features_{'otsu' if 'otsu' in key else 'grabcut'}.pkl"
        if all(os.path.exists(p) for p in [clf_path, scl_path, fts_path]):
            models[key] = {
                "clf":     joblib.load(clf_path),
                "scaler":  joblib.load(scl_path),
                "cols":    joblib.load(fts_path),
                "clf_type": clf_type,
                "seg":      seg,
                "label":    f"{clf_type} + {seg}",
            }
    # CNN
    cnn_paths = ["outputs/models/mobilenetv2_frutas.keras",
                 "outputs/modelos/mobilenetv2_frutas.h5",
                 "outputs/models/mobilenetv2_frutas.h5"]
    if CNN_AVAILABLE:
        for p in cnn_paths:
            if os.path.exists(p):
                try:
                    models["cnn"] = {"model": load_model(p), "label": "MobileNetV2"}
                    break
                except Exception:
                    pass
    return models

MODELS = load_all_models()

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🍊 FruitVision AI")
    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["🔬 Single Image", "📊 Dataset Evaluation", "📈 Feature Analysis", "🧠 XAI / SHAP"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">Models loaded</div>', unsafe_allow_html=True)
    for key, m in MODELS.items():
        dot = "dot-green"
        st.markdown(f'<span class="dot {dot}"></span>{m["label"]}', unsafe_allow_html=True)
    if not MODELS:
        st.warning("No models found.\nRun notebooks 01–03 first.")

    st.markdown("---")
    st.markdown('<div class="section-label">Dataset path</div>', unsafe_allow_html=True)
    DATASET_PATH = st.text_input("", "./dataset", label_visibility="collapsed")
    LIMIT        = st.slider("Max images / class", 20, 500, 100)

# ════════════════════════════════════════════════════════════════════════════
# UTIL — figure helpers
# ════════════════════════════════════════════════════════════════════════════

def dark_fig(w=8, h=4, **kw):
    fig, ax = plt.subplots(figsize=(w, h), **kw)
    fig.patch.set_facecolor("#0f1117")
    if isinstance(ax, np.ndarray):
        for a in ax.flat:
            a.set_facecolor("#1a1d2e")
            a.tick_params(colors="#64748b"); a.xaxis.label.set_color("#94a3b8")
            a.yaxis.label.set_color("#94a3b8"); a.title.set_color("#e2e8f0")
            for spine in a.spines.values(): spine.set_edgecolor("#2a2d3a")
    else:
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="#64748b"); ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8"); ax.title.set_color("#e2e8f0")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2d3a")
    return fig, ax


def conf_color(v):
    if v >= 0.85: return "#4ade80"
    if v >= 0.65: return "#fbbf24"
    return "#f87171"


def pred_badge(label):
    cls = "badge-fresh" if label == "fresh" else "badge-rotten"
    return f'<span class="badge {cls}">{label.upper()}</span>'


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE IMAGE
# ════════════════════════════════════════════════════════════════════════════

def page_single():
    st.markdown("# 🔬 Single Image Analysis")
    st.markdown("Upload a fruit image — all models run simultaneously with full segmentation visualization.")

    uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if not uploaded:
        st.info("Upload an image to start.")
        return

    img_pil = Image.open(uploaded).convert("RGB")
    img     = np.array(img_pil)
    img     = normalize_lighting(img)

    with st.spinner("Segmenting…"):
        t0 = time.time()
        mask_otsu = segment_otsu(img)
        t_otsu    = time.time() - t0
        t1        = time.time()
        mask_gc   = segment_grabcut(img)
        t_gc      = time.time() - t1

    feats_otsu = extract_features(img, mask_otsu)
    feats_gc   = extract_features(img, mask_gc)

    # ── Segmentation row ────────────────────────────────────────────────────
    st.markdown("### Segmentation")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-label">Original (CLAHE)</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
    with c2:
        overlay_otsu = img.copy()
        overlay_otsu[mask_otsu == 0] = [20, 20, 30]
        st.markdown(f'<div class="section-label">Otsu &nbsp;<span class="badge badge-neutral">{t_otsu*1000:.0f} ms</span></div>', unsafe_allow_html=True)
        st.image(overlay_otsu, use_container_width=True)
    with c3:
        overlay_gc = img.copy()
        overlay_gc[mask_gc == 0] = [20, 20, 30]
        st.markdown(f'<div class="section-label">GrabCut &nbsp;<span class="badge badge-neutral">{t_gc*1000:.0f} ms</span></div>', unsafe_allow_html=True)
        st.image(overlay_gc, use_container_width=True)

    st.markdown("---")
    st.markdown("### Model Predictions")

    # ── All classical models ─────────────────────────────────────────────────
    for key, m in MODELS.items():
        if key == "cnn": continue
        seg    = m["seg"]
        feats  = feats_otsu if seg == "Otsu" else feats_gc
        if feats is None:
            continue
        pred, prob = classify_rf(feats, m["clf"], m["scaler"], m["cols"])
        conf       = prob.get(pred, 0)
        classes    = sorted(prob.keys())

        with st.container():
            st.markdown(f'<div class="card card-accent">', unsafe_allow_html=True)
            ca, cb, cc = st.columns([2, 3, 3])
            with ca:
                st.markdown(f'<div class="section-label">{m["label"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="big-metric">{pred_badge(pred)}</div>', unsafe_allow_html=True)
                color = conf_color(conf)
                st.markdown(f'<span style="color:{color};font-size:1.3rem;font-weight:600">{conf:.1%}</span> confidence', unsafe_allow_html=True)
            with cb:
                st.markdown('<div class="section-label">Class probabilities</div>', unsafe_allow_html=True)
                for cls in classes:
                    p = prob.get(cls, 0)
                    bar_color = "#4ade80" if cls == "fresh" else "#f87171"
                    st.markdown(f"""
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
                          <span style="width:60px;color:#94a3b8;font-size:0.8rem">{cls}</span>
                          <div class="pred-bar-wrap" style="flex:1">
                            <div class="pred-bar" style="width:{p*100:.1f}%;background:{bar_color}"></div>
                          </div>
                          <span style="width:42px;text-align:right;color:#e2e8f0;font-size:0.8rem">{p:.1%}</span>
                        </div>""", unsafe_allow_html=True)
            with cc:
                if feats:
                    st.markdown('<div class="section-label">Key features</div>', unsafe_allow_html=True)
                    key_feats = ["mean_H","mean_S","mean_R","circularity","glcm_contrast","solidity"]
                    for kf in key_feats:
                        if kf in feats:
                            st.markdown(f'<span style="color:#64748b;font-size:0.75rem">{kf}</span> '
                                        f'<span style="color:#e2e8f0;font-size:0.8rem;float:right">{feats[kf]:.3f}</span><br>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── CNN ──────────────────────────────────────────────────────────────────
    if "cnn" in MODELS:
        model_cnn = MODELS["cnn"]["model"]
        img_r = cv2.resize(img, (224, 224))
        cnn_in = np.expand_dims(preprocess_input(img_r.astype(np.float32)), 0)
        prob_rotten = float(model_cnn.predict(cnn_in, verbose=0)[0][0])
        pred_cnn = "rotten" if prob_rotten > 0.5 else "fresh"
        conf_cnn = prob_rotten if pred_cnn == "rotten" else 1 - prob_rotten

        st.markdown(f'<div class="card card-accent">', unsafe_allow_html=True)
        ca, cb, _ = st.columns([2, 3, 3])
        with ca:
            st.markdown('<div class="section-label">MobileNetV2 (CNN)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="big-metric">{pred_badge(pred_cnn)}</div>', unsafe_allow_html=True)
            color = conf_color(conf_cnn)
            st.markdown(f'<span style="color:{color};font-size:1.3rem;font-weight:600">{conf_cnn:.1%}</span> confidence', unsafe_allow_html=True)
        with cb:
            st.markdown('<div class="section-label">Class probabilities</div>', unsafe_allow_html=True)
            for cls, p in [("fresh", 1-prob_rotten), ("rotten", prob_rotten)]:
                bar_color = "#4ade80" if cls == "fresh" else "#f87171"
                st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
                      <span style="width:60px;color:#94a3b8;font-size:0.8rem">{cls}</span>
                      <div class="pred-bar-wrap" style="flex:1">
                        <div class="pred-bar" style="width:{p*100:.1f}%;background:{bar_color}"></div>
                      </div>
                      <span style="width:42px;text-align:right;color:#e2e8f0;font-size:0.8rem">{p:.1%}</span>
                    </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def page_dataset():
    st.markdown("# 📊 Dataset Evaluation")
    st.markdown("Runs all loaded models over a folder of images and produces the full comparative report required by the project rubric.")

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        run_cnn = st.checkbox("Include CNN (slow)", value=False)
    with col_cfg2:
        show_errors = st.checkbox("Show misclassified gallery", value=True)

    if not st.button("▶ Run Evaluation", type="primary"):
        st.info("Configure options above then click **Run Evaluation**.")
        return

    # ── Collect images ───────────────────────────────────────────────────────
    label_map = {}
    for subfolder in glob.glob(os.path.join(DATASET_PATH, "**"), recursive=False):
        name = os.path.basename(subfolder).lower()
        if "fresh" in name: label_map[subfolder] = "fresh"
        elif "rotten" in name: label_map[subfolder] = "rotten"
    for subfolder in glob.glob(os.path.join(DATASET_PATH, "train", "**"), recursive=False):
        name = os.path.basename(subfolder).lower()
        if "fresh" in name: label_map[subfolder] = "fresh"
        elif "rotten" in name: label_map[subfolder] = "rotten"

    if not label_map:
        st.error("No fresh/rotten subfolders found. Check the dataset path.")
        return

    all_files = []
    for folder, lbl in label_map.items():
        files = (glob.glob(os.path.join(folder, "*.[jJ][pP]*")) +
                 glob.glob(os.path.join(folder, "*.[pP][nN][gG]")))[:LIMIT]
        all_files.extend([(f, lbl) for f in files])

    st.info(f"Found **{len(all_files)}** images across {len(label_map)} folders.")

    results      = []
    misclassified = []
    prog         = st.progress(0, text="Processing…")

    for i, (fpath, true_lbl) in enumerate(all_files):
        try:
            img = np.array(Image.open(fpath).convert("RGB"))
        except Exception:
            continue
        img = normalize_lighting(img)
        mask_o = segment_otsu(img)
        mask_g = segment_grabcut(img)
        f_o    = extract_features(img, mask_o)
        f_g    = extract_features(img, mask_g)

        row = {"true": true_lbl, "path": fpath, "img": img}

        any_wrong = False
        for key, m in MODELS.items():
            if key == "cnn": continue
            feats = f_o if m["seg"] == "Otsu" else f_g
            if feats is None:
                row[key] = None; row[f"{key}_conf"] = 0; continue
            pred, prob = classify_rf(feats, m["clf"], m["scaler"], m["cols"])
            row[key]         = pred
            row[f"{key}_conf"] = prob.get(pred, 0)
            if pred != true_lbl: any_wrong = True

        if run_cnn and "cnn" in MODELS:
            img_r  = cv2.resize(img, (224, 224))
            cnn_in = np.expand_dims(preprocess_input(img_r.astype(np.float32)), 0)
            p_rot  = float(MODELS["cnn"]["model"].predict(cnn_in, verbose=0)[0][0])
            pred_c = "rotten" if p_rot > 0.5 else "fresh"
            row["cnn"]      = pred_c
            row["cnn_conf"] = p_rot if pred_c == "rotten" else 1 - p_rot
            if pred_c != true_lbl: any_wrong = True

        results.append(row)
        if any_wrong: misclassified.append(row)
        prog.progress((i+1)/len(all_files), text=f"Processing {i+1}/{len(all_files)}…")

    prog.empty()
    df = pd.DataFrame(results)
    if df.empty:
        st.error("No results — check image paths.")
        return

    # ── Metrics table ────────────────────────────────────────────────────────
    st.markdown("### Comparative Metrics Table")
    clf_keys = [k for k in MODELS if k in df.columns]
    if run_cnn and "cnn" in MODELS and "cnn" in df.columns:
        clf_keys.append("cnn")

    rows_table = []
    best_f1 = 0
    for key in clf_keys:
        col = df[key].dropna()
        mask_valid = df[key].notna()
        y_true = df.loc[mask_valid, "true"]
        y_pred = df.loc[mask_valid, key]
        pos = "rotten"
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=pos, zero_division=0)
        rec  = recall_score(y_true, y_pred, pos_label=pos, zero_division=0)
        f1   = f1_score(y_true, y_pred, pos_label=pos, zero_division=0)
        lbl  = MODELS[key]["label"] if key in MODELS else "CNN"
        rows_table.append({"Model": lbl, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "_key": key})
        best_f1 = max(best_f1, f1)

    table_html = '<table class="styled-table"><thead><tr>'
    for col in ["Model","Accuracy","Precision","Recall","F1"]:
        table_html += f"<th>{col}</th>"
    table_html += "</tr></thead><tbody>"
    for r in rows_table:
        is_best = abs(r["F1"] - best_f1) < 1e-6
        row_cls = ' class="best-row"' if is_best else ""
        table_html += f"<tr{row_cls}>"
        table_html += f"<td>{'★ ' if is_best else ''}{r['Model']}</td>"
        for m in ["Accuracy","Precision","Recall","F1"]:
            table_html += f"<td>{r[m]:.4f}</td>"
        table_html += "</tr>"
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption("★ Best model by F1 score")

    # ── Bar chart ────────────────────────────────────────────────────────────
    st.markdown("### Visual Comparison")
    fig, ax = dark_fig(11, 4)
    x     = np.arange(len(rows_table))
    w     = 0.2
    mets  = ["Accuracy","Precision","Recall","F1"]
    colors= ["#3b82f6","#10b981","#f59e0b","#f97316"]
    for j, (met, col) in enumerate(zip(mets, colors)):
        vals = [r[met] for r in rows_table]
        ax.bar(x + j*w, vals, w, label=met, color=col, alpha=0.85)
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels([r["Model"] for r in rows_table], rotation=15, ha="right", color="#e2e8f0", fontsize=8)
    ax.set_ylim(0, 1.08); ax.legend(facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=8)
    ax.set_title("All Models — Accuracy / Precision / Recall / F1")
    st.pyplot(fig)

    # ── Confusion matrices ───────────────────────────────────────────────────
    st.markdown("### Confusion Matrices")
    n_cols = min(len(clf_keys), 4)
    cols_cm = st.columns(n_cols)
    for idx, key in enumerate(clf_keys):
        mask_valid = df[key].notna()
        y_true = df.loc[mask_valid, "true"]; y_pred = df.loc[mask_valid, key]
        cm = confusion_matrix(y_true, y_pred, labels=["fresh","rotten"])
        fig, ax = dark_fig(3.5, 3)
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                    xticklabels=["fresh","rotten"], yticklabels=["fresh","rotten"],
                    ax=ax, cbar=False, annot_kws={"size":12,"color":"white"})
        ax.set_title(MODELS[key]["label"] if key in MODELS else "CNN", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        with cols_cm[idx % n_cols]:
            st.pyplot(fig)

    # ── ROC curves ───────────────────────────────────────────────────────────
    st.markdown("### ROC Curves")
    fig, ax = dark_fig(8, 5)
    pal = ["#3b82f6","#10b981","#f59e0b","#f97316","#a78bfa"]
    for idx, key in enumerate(clf_keys):
        mask_valid = df[key].notna()
        y_true = df.loc[mask_valid, "true"]; y_pred_lbl = df.loc[mask_valid, key]
        conf_col = f"{key}_conf"
        if conf_col not in df.columns: continue
        scores = df.loc[mask_valid, conf_col].values
        y_bin  = 1 if (y_true == "rotten") else 0
        fpr, tpr, _ = roc_curve(y_bin, scores)
        roc_auc = auc(fpr, tpr)
        lbl = MODELS[key]["label"] if key in MODELS else "CNN"
        ax.plot(fpr, tpr, lw=2, color=pal[idx % len(pal)], label=f"{lbl} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1],"--", color="#334155", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=9)
    st.pyplot(fig)

    # ── Confidence distribution ───────────────────────────────────────────────
    st.markdown("### Prediction Confidence Distribution")
    fig, ax = dark_fig(9, 4)
    for idx, key in enumerate(clf_keys):
        conf_col = f"{key}_conf"
        if conf_col not in df.columns: continue
        lbl = MODELS[key]["label"] if key in MODELS else "CNN"
        sns.kdeplot(df[conf_col].dropna(), ax=ax, label=lbl, fill=True, alpha=0.25, color=pal[idx % len(pal)])
    ax.set_xlabel("Confidence"); ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution per Model")
    ax.legend(facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=9)
    st.pyplot(fig)

    # ── Error gallery ────────────────────────────────────────────────────────
    if show_errors and misclassified:
        st.markdown(f"### ❌ Misclassified Samples ({len(misclassified)} total)")
        st.caption("Showing up to 12. Common causes: stage of ripeness ambiguous, background leaking into mask, subtle early-decay texture invisible to classical features.")
        cols_err = st.columns(4)
        for i, row in enumerate(misclassified[:12]):
            preds_str = " | ".join(
                f"{MODELS[k]['label']}: {row[k]}" for k in clf_keys if k in row and row[k]
            )
            with cols_err[i % 4]:
                st.image(row["img"], use_container_width=True)
                true_badge = pred_badge(row["true"])
                st.markdown(f'True: {true_badge}<br><span style="font-size:0.7rem;color:#64748b">{preds_str}</span>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def page_features():
    st.markdown("# 📈 Feature Analysis")
    st.markdown("Loads the extracted CSVs and shows the exploratory analysis required by the project rubric (boxplots, discriminative scores, group comparison).")

    seg = st.selectbox("Segmentation method", ["Otsu", "GrabCut"])
    csv_key = "otsu" if seg == "Otsu" else "grabcut"
    x_path = f"outputs/X_{csv_key}.csv"
    y_path = f"outputs/y_{csv_key}.csv"

    if not os.path.exists(x_path):
        st.warning(f"`{x_path}` not found — run notebook 02 first.")
        return

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze()
    df = X.copy(); df["label"] = y
    st.success(f"Loaded {len(df)} samples — {dict(y.value_counts())}")

    # ── Boxplots ─────────────────────────────────────────────────────────────
    st.markdown("### Feature Distributions by Class")
    PLOT_FEATS = [c for c in ["mean_R","mean_G","mean_B","mean_H","mean_S",
                               "glcm_contrast","glcm_homogeneity","glcm_energy","glcm_correlation",
                               "circularity","solidity","eccentricity","area","perimeter"] if c in df.columns]
    n = len(PLOT_FEATS)
    n_cols_bp = 4
    n_rows_bp = (n + n_cols_bp - 1) // n_cols_bp
    fig, axes = plt.subplots(n_rows_bp, n_cols_bp, figsize=(18, n_rows_bp*3.2))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes.flat: ax.set_facecolor("#1a1d2e")
    pal_bp = {"fresh": "#4ade80", "rotten": "#f87171"}
    for i, feat in enumerate(PLOT_FEATS):
        ax = axes.flat[i]
        sns.boxplot(data=df, x="label", y=feat, ax=ax, palette=pal_bp,
                    order=["fresh","rotten"], width=0.5, linewidth=1)
        ax.set_title(feat, color="#e2e8f0", fontsize=10)
        ax.set_xlabel(""); ax.tick_params(colors="#64748b")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2d3a")
    for j in range(n, n_rows_bp*n_cols_bp):
        axes.flat[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    # ── Discriminative score ─────────────────────────────────────────────────
    st.markdown("### Discriminative Power per Feature")
    st.caption("Score = |mean_fresh − mean_rotten| / std — higher means the feature separates classes better.")

    def disc(feat):
        g = df.groupby("label")[feat].mean()
        s = df[feat].std()
        return abs(g.iloc[0] - g.iloc[1]) / (s + 1e-9)

    scores = pd.Series({f: disc(f) for f in PLOT_FEATS}).sort_values(ascending=True)
    fig, ax = dark_fig(9, max(4, n * 0.35))
    colors_bar = ["#f97316" if v == scores.max() else "#3b82f6" for v in scores.values]
    scores.plot(kind="barh", ax=ax, color=colors_bar)
    ax.set_xlabel("Discriminative Score"); ax.set_title("Feature Discriminative Power")
    for i, v in enumerate(scores.values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", color="#94a3b8", fontsize=8)
    st.pyplot(fig)

    # ── Group comparison ─────────────────────────────────────────────────────
    st.markdown("### Feature Group Comparison")
    st.caption("Compares mean discriminative score across feature families.")
    groups = {
        "Color (RGB)":     [c for c in ["mean_R","mean_G","mean_B"] if c in df.columns],
        "Color (HSV)":     [c for c in ["mean_H","mean_S"] if c in df.columns],
        "Texture (GLCM)":  [c for c in ["glcm_contrast","glcm_homogeneity","glcm_energy","glcm_correlation"] if c in df.columns],
        "Shape":           [c for c in ["circularity","solidity","eccentricity","area","aspect_ratio","extent"] if c in df.columns],
        "Hu Moments":      [c for c in df.columns if c.startswith("hu_moment_")],
    }
    group_scores = {g: np.mean([disc(f) for f in feats]) for g, feats in groups.items() if feats}
    fig, ax = dark_fig(8, 4)
    gs_series = pd.Series(group_scores).sort_values(ascending=False)
    bar_colors = ["#f97316","#3b82f6","#10b981","#f59e0b","#a78bfa"][:len(gs_series)]
    gs_series.plot(kind="bar", ax=ax, color=bar_colors, alpha=0.85)
    ax.set_ylabel("Mean Discriminative Score"); ax.set_title("Feature Group Comparison")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)

    # ── Means table ──────────────────────────────────────────────────────────
    st.markdown("### Mean Values per Class")
    means = df.groupby("label")[PLOT_FEATS].mean().round(4)
    st.dataframe(means.style.background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — XAI / SHAP
# ════════════════════════════════════════════════════════════════════════════

def page_xai():
    st.markdown("# 🧠 XAI — Model Explainability (SHAP)")
    st.markdown("SHAP (SHapley Additive exPlanations) shows which features drive each prediction, required for the advanced level of the project rubric.")

    if not SHAP_AVAILABLE:
        st.error("`shap` not installed — `pip install shap`")
        return

    seg = st.selectbox("Segmentation method", ["GrabCut","Otsu"])
    clf_type = st.selectbox("Classifier", ["RF","SVM"])
    key = f"{'rf' if clf_type=='RF' else 'svm'}_{'grabcut' if seg=='GrabCut' else 'otsu'}"

    if key not in MODELS:
        st.warning(f"Model `{key}` not loaded. Run notebook 03 first.")
        return

    m = MODELS[key]
    x_path = f"outputs/X_{'grabcut' if seg=='GrabCut' else 'otsu'}.csv"
    if not os.path.exists(x_path):
        st.warning("CSV not found — run notebook 02 first."); return

    X = pd.read_csv(x_path)
    if "Unnamed: 0" in X.columns: X = X.drop(columns=["Unnamed: 0"])

    n_sample = st.slider("SHAP sample size (larger = slower)", 50, 300, 100)

    if not st.button("▶ Compute SHAP", type="primary"):
        st.info("Click **Compute SHAP** to run (may take 10–30s).")
        return

    X_s = X.sample(n=min(n_sample, len(X)), random_state=42).reset_index(drop=True)

    with st.spinner("Computing SHAP values…"):
        explainer   = shap.TreeExplainer(m["clf"])
        shap_values = explainer.shap_values(X_s)

        if isinstance(shap_values, list):
            sv2d = shap_values[1]
        elif np.array(shap_values).ndim == 3:
            sv2d = np.array(shap_values)[:, :, 1]
        else:
            sv2d = np.array(shap_values)

        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(np.array(base_val)[1])
        else:
            base_val = float(base_val)

    st.success(f"SHAP computed on {len(X_s)} samples.")

    # ── Summary plot ────────────────────────────────────────────────────────
    st.markdown("### Global Feature Impact (Summary Plot)")
    st.caption("Each dot = one sample. Color = feature value (red=high, blue=low). X-axis = SHAP value (positive = pushes toward Rotten).")
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1d2e")
    shap.summary_plot(sv2d, X_s, show=False, plot_size=None)
    ax.tick_params(colors="#64748b"); ax.set_title("SHAP Summary — Class: Rotten", color="#e2e8f0")
    for spine in ax.spines.values(): spine.set_edgecolor("#2a2d3a")
    st.pyplot(fig)

    # ── Bar summary ──────────────────────────────────────────────────────────
    st.markdown("### Mean |SHAP| per Feature")
    mean_shap = np.abs(sv2d).mean(axis=0)
    shap_series = pd.Series(mean_shap, index=X_s.columns).sort_values(ascending=True).tail(15)
    fig, ax = dark_fig(9, 5)
    shap_series.plot(kind="barh", ax=ax, color="#f97316", alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|"); ax.set_title("Feature Importance via SHAP")
    st.pyplot(fig)

    # ── Waterfall ────────────────────────────────────────────────────────────
    st.markdown("### Local Explanation — Single Sample")
    sample_idx = st.slider("Sample index", 0, len(X_s)-1, 5)
    exp = shap.Explanation(
        values=sv2d[sample_idx],
        base_values=base_val,
        data=X_s.iloc[sample_idx].values,
        feature_names=list(X_s.columns),
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1d2e")
    shap.plots.waterfall(exp, show=False)
    ax.tick_params(colors="#64748b"); ax.set_title(f"Waterfall — Sample {sample_idx}", color="#e2e8f0")
    for spine in ax.spines.values(): spine.set_edgecolor("#2a2d3a")
    st.pyplot(fig)

    st.markdown("""
**How to interpret these plots:**

- **Summary plot:** Features at the top matter most. Red dots above 0 = high feature value pushes toward *Rotten*.
  For fruits, high `glcm_contrast` and low `mean_S` (saturation) are typical *Rotten* signals.
- **Mean |SHAP| bar:** Same as feature importance but model-agnostic and more reliable than impurity-based importance.
- **Waterfall:** Explains one specific prediction. Each bar shows how much a single feature moved the probability
  up (red) or down (blue) from the base rate. The final bar lands on the model's output.
""")


# ════════════════════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════════════════════

if   page == "🔬 Single Image":       page_single()
elif page == "📊 Dataset Evaluation": page_dataset()
elif page == "📈 Feature Analysis":   page_features()
elif page == "🧠 XAI / SHAP":        page_xai()
