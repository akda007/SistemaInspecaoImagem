import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# TensorFlow imports for the Deep Learning model
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Inspeção Visual Automática", layout="wide")
st.title("🍎 Sistema de Inspeção Visual Automática")
st.markdown("Faça o upload de uma imagem para análise. Alterne entre o pipeline clássico e a rede neural para comparar os resultados no mundo real.")

# ==========================================
# 2. CARREGAMENTO DOS MODELOS (CACHE)
# ==========================================
@st.cache_resource
def load_all_models():
    try:
        # Modelos Clássicos
        rf = joblib.load('outputs/modelos/rf_model.pkl')
        scaler = joblib.load('outputs/modelos/scaler.pkl')
        cols = joblib.load('outputs/modelos/colunas.pkl')
        
        # Modelo Deep Learning
        cnn = load_model('outputs/modelos/mobilenetv2_frutas.h5')
        
        return rf, scaler, cols, cnn
    except Exception as e:
        st.error(f"Erro ao carregar modelos. Rode os notebooks 03 e 04 primeiro. Detalhe: {e}")
        return None, None, None, None

rf_model, scaler, feature_columns, cnn_model = load_all_models()

# ==========================================
# 3. FUNÇÕES DE PROCESSAMENTO
# ==========================================
def processar_pipeline_classico(img_rgb):
    """Aplica Otsu e extrai features matemáticas manuais."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    fruta_isolada = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    
    features = {}
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, fruta_isolada, None
        
    c = max(contours, key=cv2.contourArea)
    
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    features['area'] = area
    features['perimeter'] = perimeter
    features['circularity'] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    moments = cv2.moments(c)
    hu_moments = cv2.HuMoments(moments).flatten()
    for i, hu in enumerate(hu_moments):
        features[f'hu_moment_{i}'] = -np.sign(hu) * np.log10(np.abs(hu)) if hu != 0 else 0

    mean_val = cv2.mean(img_rgb, mask=mask)
    features['mean_R'] = mean_val[0]
    features['mean_G'] = mean_val[1]
    features['mean_B'] = mean_val[2]

    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    glcm = graycomatrix(gray_masked, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features['glcm_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    features['glcm_correlation'] = graycoprops(glcm, 'correlation')[0, 0]
    features['glcm_energy'] = graycoprops(glcm, 'energy')[0, 0]
    features['glcm_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]

    return mask, fruta_isolada, features

def processar_pipeline_cnn(img_rgb):
    """Redimensiona e pré-processa a imagem para a MobileNetV2."""
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_preprocessed = preprocess_input(img_resized)
    # Expande a dimensão para simular o batch (1, 224, 224, 3)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    return img_resized, img_batch

# ==========================================
# 4. INTERFACE DO USUÁRIO
# ==========================================
# Painel lateral para configurações
st.sidebar.header("⚙️ Configurações do Motor")
motor_escolhido = st.sidebar.radio(
    "Escolha a Arquitetura de IA:",
    ("1. Pipeline Clássico (Random Forest)", "2. Deep Learning (MobileNetV2)")
)

uploaded_file = st.file_uploader("Upload da imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and rf_model is not None and cnn_model is not None:
    image_pil = Image.open(uploaded_file)
    img_rgb = np.array(image_pil)
    
    st.markdown("---")
    
    # ROTA 1: PIPELINE CLÁSSICO
    if motor_escolhido == "1. Pipeline Clássico (Random Forest)":
        st.subheader("Motor: Visão Computacional Clássica")
        mask, fruta_isolada, features = processar_pipeline_classico(img_rgb)
        
        if features is None:
            st.warning("Falha na segmentação de Otsu: Nenhum objeto detectado.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img_rgb, caption='1. Imagem Original', use_column_width=True)
            with col2:
                st.image(mask, caption='2. Segmentação (Falha em fundos complexos)', use_column_width=True, clamp=True, channels='GRAY')
            with col3:
                st.image(fruta_isolada, caption='3. Objeto Isolado', use_column_width=True)

            df_features = pd.DataFrame([features])[feature_columns]
            scaled_features = scaler.transform(df_features)
            
            prediction = rf_model.predict(scaled_features)[0]
            probabilidades = rf_model.predict_proba(scaled_features)[0]
            confianca = max(probabilidades) * 100
            
            st.markdown("---")
            if prediction == 'fresh':
                st.success(f"✅ Veredito Clássico: FRESH (Fresco) | Confiança: {confianca:.1f}%")
            else:
                st.error(f"❌ Veredito Clássico: ROTTEN (Podre) | Confiança: {confianca:.1f}%")

            with st.expander("Ver Matriz de Features Matemáticas"):
                df_display = df_features.T.reset_index()
                df_display.columns = ['Feature', 'Valor Extraído']
                st.dataframe(df_display, use_container_width=True)

    # ROTA 2: DEEP LEARNING (TRANSFER LEARNING)
    else:
        st.subheader("Motor: Rede Neural Convolucional (MobileNetV2)")
        img_resized, img_batch = processar_pipeline_cnn(img_rgb)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption='1. Imagem Original', use_column_width=True)
        with col2:
            st.image(img_resized, caption='2. Resize para 224x224 (Sem corte de fundo)', use_column_width=True)

        probabilidade_rotten = cnn_model.predict(img_batch)[0][0]
        
        st.markdown("---")
        # Se a probabilidade de ser classe 1 (Rotten) for maior que 50%
        if probabilidade_rotten > 0.5:
            confianca = probabilidade_rotten * 100
            st.error(f"❌ Veredito CNN: ROTTEN (Podre) | Confiança: {confianca:.1f}%")
        else:
            confianca = (1 - probabilidade_rotten) * 100
            st.success(f"✅ Veredito CNN: FRESH (Fresco) | Confiança: {confianca:.1f}%")
            
        st.info("💡 Note que a CNN não depende da segmentação do fundo (Otsu) nem de cálculos manuais de área/cor. Ela processa os pixels diretamente, tornando-a muito mais robusta para imagens do mundo real.")