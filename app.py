import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import os

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import graycomatrix, graycoprops

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="Apple AI System", layout="wide")
st.title("🍎 Apple Quality AI System (Fixed Pipeline)")

# ==================================================
# LOAD MODELS
# ==================================================
@st.cache_resource
def load_models():

    rf_otsu = joblib.load("outputs/models/rf_otsu.pkl")
    scaler_otsu = joblib.load("outputs/models/scaler_otsu.pkl")
    cols_otsu = joblib.load("outputs/models/features_otsu.pkl")

    rf_gc = joblib.load("outputs/models/rf_grabcut.pkl")
    scaler_gc = joblib.load("outputs/models/scaler_grabcut.pkl")
    cols_gc = joblib.load("outputs/models/features_grabcut.pkl")

    cnn = load_model("outputs/modelos/mobilenetv2_frutas.h5")

    return rf_otsu, scaler_otsu, cols_otsu, rf_gc, scaler_gc, cols_gc, cnn


(
    rf_otsu,
    scaler_otsu,
    cols_otsu,
    rf_gc,
    scaler_gc,
    cols_gc,
    cnn_model
) = load_models()

# ==================================================
# NAVIGATION
# ==================================================
page = st.sidebar.radio(
    "Navigation",
    ["Single Image Demo", "Dataset Evaluation"]
)

# ==================================================
# PREPROCESSING
# ==================================================
def normalize(img):

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge([l,a,b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def keep_largest(mask):

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if n <= 1:
        return mask

    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    return (labels == biggest).astype(np.uint8) * 255

# ==================================================
# SEGMENTATION
# ==================================================
def otsu(img):

    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (7,7), 0)

    _, m = cv2.threshold(
        g, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5,5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    return keep_largest(m)


def grabcut(img):

    h,w = img.shape[:2]

    mask = np.zeros((h,w), np.uint8)

    rect = (
        int(w*0.15),
        int(h*0.15),
        int(w*0.7),
        int(h*0.7)
    )

    bg = np.zeros((1,65), np.float64)
    fg = np.zeros((1,65), np.float64)

    cv2.grabCut(
        img, mask, rect,
        bg, fg, 5,
        cv2.GC_INIT_WITH_RECT
    )

    m = np.where(
        (mask==2)|(mask==0),
        0,255
    ).astype(np.uint8)

    return keep_largest(m)

# ==================================================
# 🔥 FULL FEATURE EXTRACTOR (MATCHES TRAINING)
# ==================================================
def extract_features(img, mask):

    contours,_ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)

    features = {}

    # -------------------------
    # Shape features
    # -------------------------
    area = cv2.contourArea(c)
    per = cv2.arcLength(c, True)

    features["area"] = area
    features["perimeter"] = per
    features["circularity"] = (4*np.pi*area)/(per**2+1e-6)

    x,y,w,h = cv2.boundingRect(c)
    features["aspect_ratio"] = w/(h+1e-6)
    features["extent"] = area/(w*h+1e-6)

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    features["solidity"] = area/(hull_area+1e-6)

    # -------------------------
    # Hu moments
    # -------------------------
    m = cv2.moments(c)
    hu = cv2.HuMoments(m).flatten()

    for i,v in enumerate(hu):
        features[f"hu_moment_{i}"] = -np.sign(v)*np.log10(abs(v)+1e-6)

    # -------------------------
    # Color features
    # -------------------------
    mean = cv2.mean(img, mask=mask)
    features["mean_R"] = mean[0]
    features["mean_G"] = mean[1]
    features["mean_B"] = mean[2]

    # -------------------------
    # Texture (GLCM SAFE)
    # -------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    pixels = gray[mask > 0]

    if len(pixels) > 1000:

        size = int(np.sqrt(len(pixels)))
        pixels = pixels[:size*size]
        tex = pixels.reshape(size, size)

        glcm = graycomatrix(
            tex,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )

        features["glcm_contrast"] = graycoprops(glcm,"contrast")[0,0]
        features["glcm_correlation"] = graycoprops(glcm,"correlation")[0,0]
        features["glcm_energy"] = graycoprops(glcm,"energy")[0,0]
        features["glcm_homogeneity"] = graycoprops(glcm,"homogeneity")[0,0]

    else:
        features["glcm_contrast"] = 0
        features["glcm_correlation"] = 0
        features["glcm_energy"] = 0
        features["glcm_homogeneity"] = 0

    return features

# ==================================================
# CNN
# ==================================================
def prep_cnn(img):

    r = cv2.resize(img,(224,224))
    return r, np.expand_dims(preprocess_input(r),0)

# ==================================================
# SINGLE IMAGE PAGE
# ==================================================
def single_page():

    st.header("Single Image Test")

    file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

    if file:

        img = np.array(Image.open(file).convert("RGB"))
        img = normalize(img)

        m1 = otsu(img)
        m2 = grabcut(img)

        f1 = extract_features(img,m1)
        f2 = extract_features(img,m2)

        cnn_img, cnn_in = prep_cnn(img)

        col1,col2,col3 = st.columns(3)

        # OTSU
        with col1:
            st.subheader("Otsu RF")
            st.image(img)
            st.image(m1)

            if f1:
                df = pd.DataFrame([f1]).reindex(columns=cols_otsu, fill_value=0)

                pred = rf_otsu.predict(scaler_otsu.transform(df))[0]
                conf = max(rf_otsu.predict_proba(scaler_otsu.transform(df))[0])*100

                st.metric(pred, f"{conf:.1f}%")

        # GRABCUT
        with col2:
            st.subheader("GrabCut RF")
            st.image(img)
            st.image(m2)

            if f2:
                df = pd.DataFrame([f2]).reindex(columns=cols_gc, fill_value=0)

                pred = rf_gc.predict(scaler_gc.transform(df))[0]
                conf = max(rf_gc.predict_proba(scaler_gc.transform(df))[0])*100

                st.metric(pred, f"{conf:.1f}%")

        # CNN
        with col3:
            st.subheader("CNN")
            st.image(img)
            st.image(cnn_img)

            p = cnn_model.predict(cnn_in, verbose=0)[0][0]

            if p > 0.5:
                st.metric("rotten", f"{p*100:.1f}%")
            else:
                st.metric("fresh", f"{(1-p)*100:.1f}%")

# ==================================================
# DATASET EVALUATION PAGE (SAFE)
# ==================================================
def dataset_page():

    import matplotlib.pyplot as plt
    import seaborn as sns

    st.header("📊 Dataset Evaluation Dashboard")

    path = st.text_input("Dataset path", "../dataset")
    limit = st.slider("Max images per run", 10, 300, 80)

    if st.button("Run Evaluation"):

        results = []
        misclassified = []

        count = 0

        for root, _, files in os.walk(path):

            for f in files:

                if not f.lower().endswith(("jpg","jpeg","png")):
                    continue

                try:
                    img = np.array(Image.open(os.path.join(root,f)).convert("RGB"))
                except:
                    continue

                img = normalize(img)

                m1 = otsu(img)
                m2 = grabcut(img)

                f1 = extract_features(img, m1)
                f2 = extract_features(img, m2)

                true_label = "fresh" if "Good" in root else "rotten"

                # ------------------------
                # OTSU
                # ------------------------
                if f1:
                    df = pd.DataFrame([f1]).reindex(columns=cols_otsu, fill_value=0)
                    pred_otsu = rf_otsu.predict(scaler_otsu.transform(df))[0]
                    conf_otsu = max(rf_otsu.predict_proba(scaler_otsu.transform(df))[0])
                else:
                    pred_otsu = "error"
                    conf_otsu = 0

                # ------------------------
                # GRABCUT
                # ------------------------
                if f2:
                    df = pd.DataFrame([f2]).reindex(columns=cols_gc, fill_value=0)
                    pred_gc = rf_gc.predict(scaler_gc.transform(df))[0]
                    conf_gc = max(rf_gc.predict_proba(scaler_gc.transform(df))[0])
                else:
                    pred_gc = "error"
                    conf_gc = 0

                results.append({
                    "true": true_label,
                    "otsu": pred_otsu,
                    "grabcut": pred_gc,
                    "conf_otsu": conf_otsu,
                    "conf_grabcut": conf_gc,
                    "img": img
                })

                # track mistakes
                if pred_otsu != true_label or pred_gc != true_label:
                    misclassified.append({
                        "img": img,
                        "true": true_label,
                        "otsu": pred_otsu,
                        "grabcut": pred_gc
                    })

                count += 1
                if count >= limit:
                    break

            if count >= limit:
                break

        df = pd.DataFrame(results)

        if len(df) == 0:
            st.warning("No valid images found.")
            return

        # ==================================================
        # 📊 1. ACCURACY COMPARISON
        # ==================================================
        st.subheader("📊 Model Accuracy Comparison")

        acc_otsu = np.mean(df["otsu"] == df["true"])
        acc_gc = np.mean(df["grabcut"] == df["true"])

        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(
            x=["Otsu RF", "GrabCut RF"],
            y=[acc_otsu, acc_gc],
            palette="viridis",
            ax=ax
        )

        ax.set_ylim(0,1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Comparison")

        st.pyplot(fig)

        # ==================================================
        # 📊 2. CONFUSION MATRICES
        # ==================================================
        st.subheader("🔥 Confusion Matrices")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Otsu")
            cm1 = confusion_matrix(df["true"], df["otsu"])
            fig, ax = plt.subplots()
            sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["fresh","rotten"],
                        yticklabels=["fresh","rotten"])
            ax.set_title("Otsu RF")
            st.pyplot(fig)

        with col2:
            st.write("GrabCut")
            cm2 = confusion_matrix(df["true"], df["grabcut"])
            fig, ax = plt.subplots()
            sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens",
                        xticklabels=["fresh","rotten"],
                        yticklabels=["fresh","rotten"])
            ax.set_title("GrabCut RF")
            st.pyplot(fig)

        # ==================================================
        # 📊 3. CONFIDENCE DISTRIBUTION
        # ==================================================
        st.subheader("📈 Prediction Confidence Distribution")

        fig, ax = plt.subplots()
        sns.kdeplot(df["conf_otsu"], label="Otsu", fill=True)
        sns.kdeplot(df["conf_grabcut"], label="GrabCut", fill=True)

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)

        # ==================================================
        # 📊 4. CLASS DISTRIBUTION
        # ==================================================
        st.subheader("📦 Class Distribution")

        fig, ax = plt.subplots()
        sns.countplot(x=df["true"], palette="pastel", ax=ax)
        ax.set_title("True Labels Distribution")
        st.pyplot(fig)

        # ==================================================
        # ❌ 5. MISCLASSIFIED SAMPLES GALLERY
        # ==================================================
        st.subheader("❌ Misclassified Samples")

        if len(misclassified) == 0:
            st.success("No misclassifications 🎉")
        else:
            cols = st.columns(3)

            for i, item in enumerate(misclassified[:9]):

                with cols[i % 3]:
                    st.image(item["img"], caption=f"""
True: {item['true']}
Otsu: {item['otsu']}
GrabCut: {item['grabcut']}
""", use_column_width=True)


# ==================================================
# ROUTER
# ==================================================
if page == "Single Image Demo":
    single_page()
else:
    dataset_page()