import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 
import matplotlib.pyplot as plt
import time
import json

# -----------------
# 1. Configuración de la Página
# -----------------

st.set_page_config(
    page_title="Detección de Moniliasis en Cacao",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------
# 2. Configuración y Carga de Nombres de Clases
# -----------------

UMBRAL_CONFIANZA_NO_CACO = 70.0
MODEL_PATH = "cacao_resnet101_classifier3.keras"

try:
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
except FileNotFoundError:
    st.error("Error: El archivo 'class_names.json' no se encontró. Asegúrate de que esté en la misma carpeta que app.py.")
    st.stop()

# -----------------
# 3. Funciones de Preprocesamiento y Saliency Map
# -----------------

@st.cache_resource
def load_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo. Verifique la ruta y el formato. Error: {e}")
        st.stop()

def preprocess_image_for_prediction_resnet(image_array, target_size=(224, 224)):
    img_resized = Image.fromarray(image_array.astype(np.uint8)).resize(target_size)
    img_array_resized = np.asarray(img_resized, dtype=np.float32)
    img_preprocessed = tf.keras.applications.resnet_v2.preprocess_input(np.expand_dims(img_array_resized, axis=0))
    return img_preprocessed

def extract_numerical_features_for_prediction(image_stream):
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    percentage_black = (np.sum(black_mask > 0) / np.prod(img_bgr.shape[:2])) * 100

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(black_mask, 8, cv2.CV_32S)
    large_black_spots_count = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 100)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    mean_green_value = np.mean(hsv[:, :, 2][green_mask > 0]) if np.sum(green_mask > 0) > 0 else 0

    max_percentage_black = 100.0
    max_spots = 20
    max_green_value = 255.0

    scaled_percentage_black = percentage_black / max_percentage_black
    scaled_large_black_spots_count = min(large_black_spots_count / max_spots, 1.0)
    scaled_mean_green_value = mean_green_value / max_green_value

    return np.array([scaled_percentage_black, scaled_large_black_spots_count, scaled_mean_green_value], dtype=np.float32)

def get_prediction_info(model, image_path, preprocess_img_fn, extract_features_fn, confidence_threshold):
    original_img_pil = Image.open(image_path).convert('RGB')
    original_img_array = np.asarray(original_img_pil)

    preprocessed_img_for_resnet = preprocess_img_fn(original_img_array)
    img_tensor = tf.Variable(preprocessed_img_for_resnet, dtype=tf.float32)

    image_path.seek(0)
    numerical_features = extract_features_fn(image_path)
    numerical_features_batch = np.expand_dims(numerical_features, axis=0)
    numerical_features_tensor = tf.convert_to_tensor(numerical_features_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model([img_tensor, numerical_features_tensor])
        predicted_class_idx = tf.argmax(predictions[0])
        predicted_class_score = predictions[0, predicted_class_idx]
    
    gradients = tape.gradient(predicted_class_score, img_tensor)
    gradients = gradients[0].numpy()
    saliency_map = np.sum(np.abs(gradients), axis=-1)
    saliency_map = np.maximum(saliency_map, 0)
    if np.max(saliency_map) > 0:
        saliency_map /= np.max(saliency_map)

    saliency_map_colored = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
    saliency_map_colored = cv2.cvtColor(saliency_map_colored, cv2.COLOR_BGR2RGB)
    saliency_map_resized = cv2.resize(saliency_map_colored, (original_img_array.shape[1], original_img_array.shape[0]))
    overlayed_image = cv2.addWeighted(original_img_array.astype(np.uint8), 0.6, saliency_map_resized, 0.4, 0)
    
    confidence = predictions[0, predicted_class_idx].numpy() * 100
    predicted_class_name_raw = class_names[predicted_class_idx.numpy()]
    
    if confidence < UMBRAL_CONFIANZA_NO_CACO:
        final_prediction_text = f"NO ES UNA MAZORCA DE CACAO. (Confianza: {confidence:.2f}%)"
        display_title_text = "NO MAZORCA"
    else:
        final_prediction_text = f"Es una mazorca de cacao: {predicted_class_name_raw} (Confianza: {confidence:.2f}%)"
        display_title_text = f"{predicted_class_name_raw}"
    
    return overlayed_image, final_prediction_text, display_title_text

# -----------------
# 4. Diseño de la Interfaz y Lógica Principal
# -----------------

# Inicializa el estado de la sesión
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Barra lateral para la carga de archivos
with st.sidebar:
    st.header("Sube tu imagen")
    temp_uploaded_file = st.file_uploader("Elige una imagen de una mazorca...", type=["jpg", "png", "jpeg"])
    
    if temp_uploaded_file:
        st.session_state.uploaded_file = temp_uploaded_file
        if st.button("Analizar Imagen"):
            st.session_state.analyzed = True
    
    if temp_uploaded_file is None and st.session_state.uploaded_file is not None:
        st.session_state.analyzed = False
        st.session_state.uploaded_file = None
        st.rerun()

if not st.session_state.analyzed:
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Detección de Moniliasis en Cacao</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #666;'>Analiza tus mazorcas con inteligencia artificial y visualiza las áreas clave.</h4>", unsafe_allow_html=True)
    st.markdown("---")
    st.info("Sube una imagen desde la barra lateral para comenzar el análisis.")

elif st.session_state.analyzed and st.session_state.uploaded_file:
    st.markdown("<h1 style='text-align: center;'>Análisis y Resultados</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns(2)

    with col1:
        # Título "Imagen Subida"
        st.markdown("<h3 style='text-align: center;'>Imagen Subida</h3>", unsafe_allow_html=True)
        image = Image.open(st.session_state.uploaded_file)
        st.image(image, width=400)

    with col2:
        with st.spinner("Analizando la imagen..."):
            time.sleep(1)
            try:
                overlayed_img, final_pred_text, display_title = get_prediction_info(
                    load_model(MODEL_PATH),
                    st.session_state.uploaded_file,
                    preprocess_image_for_prediction_resnet,
                    extract_numerical_features_for_prediction,
                    UMBRAL_CONFIANZA_NO_CACO
                )
                
                # Muestra el resultado de la predicción como un encabezado h4
                st.markdown(f"<h5 style='text-align: center;'>{final_pred_text}</h5>", unsafe_allow_html=True)

                # Título "Mapa de Saliency"
                st.markdown("<h5 style='text-align: center;'>Mapa de Saliency</h5>", unsafe_allow_html=True)
                fig_sal, ax_sal = plt.subplots(figsize=(4, 4))
                ax_sal.imshow(overlayed_img)
                ax_sal.set_title(f'Saliency: {display_title}', color='#333')
                ax_sal.axis('off')
                st.pyplot(fig_sal)

            except Exception as e:
                st.error(f"Ocurrió un error inesperado durante el análisis: {e}")