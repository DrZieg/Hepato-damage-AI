import streamlit as st
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color, filters, measure
import tensorflow as tf
import random

# --- Cargar modelos al inicio de la aplicación para que no se recarguen con cada interacción ---
@st.cache_resource
def load_my_models():
    cell_model = None
    tissue_model = None
    try:
        cell_model = tf.keras.models.load_model('cell.h5') # <--- RUTA DE TU MODELO DE CÉLULAS
        st.success("Modelo de células 'cell.h5' cargado exitosamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo de células 'cell.h5': {e}. Asegúrate de que la ruta sea correcta.")

    try:
        tissue_model = tf.keras.models.load_model('tissue.h5') # <--- RUTA DE TU MODELO DE TEJIDO
        st.success("Modelo de tejido 'tissue.h5' cargado exitosamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo de tejido 'tissue.h5': {e}. Asegúrate de que la ruta sea correcta.")
    
    return cell_model, tissue_model

loaded_model_h5, loaded_tissue_model = load_my_models()

# --- Constantes y etiquetas de clase ---
class_labels = ['Sana', 'Dañada'] # <--- AJUSTA ESTO A TUS ETIQUETAS REALES
num_classes = len(class_labels)

# --- Inicializar st.session_state para persistir datos ---
if 'img_original_display' not in st.session_state:
    st.session_state.img_original_display = None
if 'gray_img_display' not in st.session_state:
    st.session_state.gray_img_display = None
if 'last_uploaded_file_name' not in st.session_state:
    st.session_state.last_uploaded_file_name = None

if 'segments_rgb_all' not in st.session_state:
    st.session_state.segments_rgb_all = np.empty((0, 48, 48, 3), dtype=np.uint8)
if 'cells_for_prediction_normalized' not in st.session_state:
    st.session_state.cells_for_prediction_normalized = np.empty((0, 96, 96, 3), dtype=np.float32)
if 'predicted_classes_cells_individual' not in st.session_state:
    st.session_state.predicted_classes_cells_individual = np.array([], dtype=int)
if 'segment_ids_processed' not in st.session_state:
    st.session_state.segment_ids_processed = []

if 'grid_segments_224x224_array' not in st.session_state:
    st.session_state.grid_segments_224x224_array = np.empty((0, 224, 224, 3), dtype=np.uint8)
if 'tissue_for_prediction_normalized' not in st.session_state:
    st.session_state.tissue_for_prediction_normalized = np.empty((0, 224, 224, 3), dtype=np.float32)
if 'predicted_classes_tissue_individual' not in st.session_state:
    st.session_state.predicted_classes_tissue_individual = np.array([], dtype=int)

if 'watershed_markers' not in st.session_state:
    st.session_state.watershed_markers = None

if 'combined_average_probabilities' not in st.session_state:
    st.session_state.combined_average_probabilities = np.zeros(num_classes)
if 'overall_combined_predicted_class' not in st.session_state:
    st.session_state.overall_combined_predicted_class = 0
if 'show_saliency' not in st.session_state:
    st.session_state.show_saliency = False


# --- Funciones de tu pipeline (adaptadas para Streamlit y cacheables) ---

@st.cache_data(show_spinner="Cargando y preprocesando imagen...")
def load_and_preprocess_original_image(uploaded_file, target_width=2000):
    if uploaded_file is None:
        return None, None

    # PIL puede abrir una amplia variedad de formatos
    rgb_image = Image.open(uploaded_file)
    
    # Convertir a RGB si no lo está (algunos PNG/JPG pueden ser RGBA o escala de grises)
    if rgb_image.mode != 'RGB':
        rgb_image = rgb_image.convert('RGB')

    original_img_np = np.array(rgb_image)

    current_height, current_width, _ = original_img_np.shape
    img_to_process = original_img_np

    if current_width > target_width:
        aspect_ratio = target_width / current_width
        target_height = int(current_height * aspect_ratio)
        resized_img_np = cv.resize(original_img_np, (target_width, target_height), interpolation=cv.INTER_AREA)
        img_to_process = resized_img_np
        st.sidebar.info(f"Imagen reescalada de {current_width}x{current_height} a {target_width}x{target_height}.")
    else:
        st.sidebar.info(f"La imagen original ya tiene {current_width}px de ancho, no se reescala.")

    gray = cv.cvtColor(img_to_process, cv.COLOR_RGB2GRAY)
    return img_to_process, gray

@st.cache_data(show_spinner=False)
def get_binary_feedback_image(gray_img, threshold_min, threshold_max, kernel_size, iterations):
    if gray_img is None:
        return None

    ret, thresh = cv.threshold(gray_img, threshold_min, threshold_max, cv.THRESH_BINARY_INV)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=iterations)
    
    return opening

@st.cache_data(show_spinner="Aplicando Watershed y extrayendo segmentos de células...")
def apply_watershed_and_extract_cells(img_original_clean, gray_img, maxt, mint, ks, it, padding=30):
    if img_original_clean is None or gray_img is None:
        return np.empty((0, 48, 48, 3), dtype=np.uint8), [], None

    ret, thresh = cv.threshold(gray_img, mint, maxt, cv.THRESH_BINARY_INV)
    kernel = np.ones((ks, ks), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=it)
    dilated_opening_for_watershed = cv.dilate(opening, kernel, iterations=15)
    sure_bg = cv.dilate(dilated_opening_for_watershed, kernel, iterations=2)
    dist_transform = cv.distanceTransform(dilated_opening_for_watershed, cv.DIST_L2, 3)
    ret, sure_fg = cv.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    num_labels, markers_initial = cv.connectedComponents(sure_fg)
    markers_initial = markers_initial + 1
    markers_initial[unknown == 255] = 0
    img_for_watershed_processing = img_original_clean.copy()
    markers = cv.watershed(img_for_watershed_processing, markers_initial)

    segments_rgb_all_list = []
    segment_ids_processed = []

    all_unique_segment_ids = [segment_id for segment_id in np.unique(markers) if segment_id != -1 and segment_id != 1]

    for segment_id in all_unique_segment_ids:
        mask = (markers == segment_id).astype(np.uint8) * 255
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0: continue
        largest_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(largest_contour)
        
        x_expanded = x - padding
        y_expanded = y - padding
        w_expanded = w + (2 * padding)
        h_expanded = h + (2 * padding)
        
        x_min_cropped = max(0, x_expanded)
        y_min_cropped = max(0, y_expanded)
        x_max_cropped = min(img_original_clean.shape[1], x_expanded + w_expanded)
        y_max_cropped = min(img_original_clean.shape[0], y_expanded + h_expanded)
        
        segment_cropped_square = img_original_clean[y_min_cropped:y_max_cropped, x_min_cropped:x_max_cropped]

        if segment_cropped_square.shape[0] == 0 or segment_cropped_square.shape[1] == 0: continue

        segment_resized = cv.resize(segment_cropped_square, (48, 48), interpolation=cv.INTER_AREA)
        
        # Sharpening
        segment_float = segment_resized.astype(np.float32) / 255.0
        blurred = cv.GaussianBlur(segment_float, (0, 0), 2.0)
        alpha_sharpen = 0.8
        sharpened_segment = cv.addWeighted(segment_float, 1.0 + alpha_sharpen, blurred, -alpha_sharpen, 0)
        sharpened_segment_uint8 = (np.clip(sharpened_segment, 0, 1) * 255).astype(np.uint8)

        segment_rgb_final = sharpened_segment_uint8
        
        segments_rgb_all_list.append(np.expand_dims(segment_rgb_final, axis=0))
        segment_ids_processed.append(segment_id)

    if segments_rgb_all_list:
        segments_rgb_all = np.concatenate(segments_rgb_all_list, axis=0)
    else:
        segments_rgb_all = np.empty((0, 48, 48, 3), dtype=np.uint8)
    
    st.info(f"Watershed: Se han extraído {segments_rgb_all.shape[0]} segmentos de células.")
    return segments_rgb_all, segment_ids_processed, markers

@st.cache_data(show_spinner="Extrayendo parches de tejido (512x512)...")
def extract_tissue_patches(img_to_process, segment_size_512=512, segment_size_224=224):
    if img_to_process is None:
        return np.empty((0, 224, 224, 3), dtype=np.uint8)

    img_height, img_width, _ = img_to_process.shape
    final_segments_224x224_list = []

    for y_start in range(0, img_height - segment_size_512 + 1, segment_size_512):
        for x_start in range(0, img_width - segment_size_512 + 1, segment_size_512):
            y_end = y_start + segment_size_512
            x_end = x_start + segment_size_512
            segment_512x512 = img_to_process[y_start:y_end, x_start:x_end]
            
            if segment_512x512.shape[0] == segment_size_512 and segment_512x512.shape[1] == segment_size_512:
                segment_224x224 = cv.resize(segment_512x512, (segment_size_224, segment_size_224), interpolation=cv.INTER_AREA)
                final_segments_224x224_list.append(segment_224x224)
    
    if final_segments_224x224_list:
        grid_segments_224x224_array = np.array(final_segments_224x224_list)
    else:
        grid_segments_224x224_array = np.empty((0, 224, 224, 3), dtype=np.uint8)

    st.info(f"Cuadrícula: Se han extraído {grid_segments_224x224_array.shape[0]} segmentos de tejido de {segment_size_224}x{segment_size_224}.")
    return grid_segments_224x224_array

@st.cache_data(show_spinner="Clasificando segmentos de células...")
def classify_cells_and_average(segments_rgb_all, loaded_model_h5, class_labels):
    if loaded_model_h5 is None or segments_rgb_all.shape[0] == 0:
        return np.zeros(len(class_labels)), np.array([], dtype=int), np.empty((0, 96, 96, 3), dtype=np.float32)

    cell_target_size = (96, 96)
    if segments_rgb_all.shape[1:3] != cell_target_size:
        resized_segments = np.array([cv.resize(img_segment, cell_target_size, interpolation=cv.INTER_AREA) for img_segment in segments_rgb_all])
    else:
        resized_segments = segments_rgb_all

    segments_normalized = resized_segments.astype('float32') / 255.0
    predictions = loaded_model_h5.predict(segments_normalized, verbose=0)
    average_probabilities = np.mean(predictions, axis=0)
    predicted_classes = np.argmax(predictions, axis=1)
    return average_probabilities, predicted_classes, segments_normalized

@st.cache_data(show_spinner="Clasificando segmentos de tejido...")
def classify_tissue_and_average(grid_segments_224x224_array, loaded_tissue_model, class_labels):
    if loaded_tissue_model is None or grid_segments_224x224_array.shape[0] == 0:
        return np.zeros(len(class_labels)), np.array([], dtype=int), np.empty((0, 224, 224, 3), dtype=np.float32)
    
    tissue_segments_normalized = grid_segments_224x224_array.astype('float32') / 255.0
    tissue_predictions = loaded_tissue_model.predict(tissue_segments_normalized, verbose=0)
    average_tissue_probabilities = np.mean(tissue_predictions, axis=0)
    tissue_predicted_classes = np.argmax(tissue_predictions, axis=1)
    return average_tissue_probabilities, tissue_predicted_classes, tissue_segments_normalized

def compute_saliency_map(model, image_normalized, class_idx):
    img_tensor = tf.convert_to_tensor(np.expand_dims(image_normalized, axis=0))
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        score = predictions[0, class_idx]

    gradients = tape.gradient(score, img_tensor)[0]
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-8)
    return saliency.numpy()

# --- Función para superponer el mapa de saliencia en la imagen original (ajustada para relieve) ---
# AJUSTE 1 (alpha) y AJUSTE 4 (smooth_saliency_kernel y cmap)
def overlay_saliency_map(image_original, saliency_map, alpha=0.5, cmap='jet', saliency_boost_factor=1.5, smooth_saliency_kernel=(9,9)):
    """
    Superpone un mapa de saliencia (escala de grises) sobre una imagen RGB.
    `alpha`: Transparencia del mapa de saliencia. Un valor más bajo muestra más el fondo.
    `cmap`: Mapa de color para la saliencia. 'jet' y 'magma' son buenos para relieve.
    `saliency_boost_factor`: Multiplicador aplicado a los valores de saliencia para exagerar.
    `smooth_saliency_kernel`: Tupla (ancho, alto) para el kernel de desenfoque gaussiano, para suavizar.
    """
    if saliency_map.shape[:2] != image_original.shape[:2]:
        saliency_map = cv.resize(saliency_map, (image_original.shape[1], image_original.shape[0]), interpolation=cv.INTER_LINEAR)
    
    # --- Suavizar el mapa de saliencia para el efecto de relieve ---
    if smooth_saliency_kernel and smooth_saliency_kernel[0] > 0 and smooth_saliency_kernel[1] > 0:
        saliency_map = cv.GaussianBlur(saliency_map, smooth_saliency_kernel, 0)
        # Re-normalizar después del suavizado
        saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map) + 1e-8)

    boosted_saliency_map = np.clip(saliency_map * saliency_boost_factor, 0, 1)

    cmap_obj = plt.cm.get_cmap(cmap)
    heatmap = cmap_obj(boosted_saliency_map)
    heatmap = (heatmap[..., :3] * 255).astype(np.uint8)
    
    image_float = image_original.astype(np.float32)
    heatmap_float = heatmap.astype(np.float32)

    overlayed_image = cv.addWeighted(image_float, 1 - alpha, heatmap_float, alpha, 0)
    return np.clip(overlayed_image, 0, 255).astype(np.uint8)


# --- Interfaz de usuario de Streamlit ---
st.title("Determinación de daño hepático por IA (Células y Tejido)")
st.sidebar.header("Controles")

# AJUSTE 2: Ampliar los tipos de archivo permitidos
uploaded_file = st.sidebar.file_uploader("Cargar imagen", type=["tif", "tiff", "jpg", "jpeg", "png", "bmp"])

# Lógica para cargar/procesar la imagen base y almacenar en session_state
if uploaded_file is not None:
    # Solo cargar y preprocesar si el archivo es nuevo o si no hay imagen en el estado
    if st.session_state.img_original_display is None or uploaded_file.name != st.session_state.get('last_uploaded_file_name'):
        img_to_process_temp, gray_img_temp = load_and_preprocess_original_image(uploaded_file)
        st.session_state.img_original_display = img_to_process_temp
        st.session_state.gray_img_display = gray_img_temp
        st.session_state.last_uploaded_file_name = uploaded_file.name
        
        # Resetear estado de procesamiento si se carga nueva imagen para evitar mostrar resultados antiguos
        st.session_state.show_saliency = False
        st.session_state.segments_rgb_all = np.empty((0, 48, 48, 3), dtype=np.uint8)
        st.session_state.cells_for_prediction_normalized = np.empty((0, 96, 96, 3), dtype=np.float32)
        st.session_state.predicted_classes_cells_individual = np.array([], dtype=int)
        st.session_state.segment_ids_processed = []
        st.session_state.grid_segments_224x224_array = np.empty((0, 224, 224, 3), dtype=np.uint8)
        st.session_state.tissue_for_prediction_normalized = np.empty((0, 224, 224, 3), dtype=np.float32)
        st.session_state.predicted_classes_tissue_individual = np.array([], dtype=int)
        st.session_state.watershed_markers = None
        st.session_state.combined_average_probabilities = np.zeros(num_classes)
        st.session_state.overall_combined_predicted_class = 0
        st.cache_data.clear() # Limpiar cache de datos para nuevo archivo y forzar re-ejecución si cambian sliders

    # Mostrar la imagen cargada/procesada en la barra lateral
    st.sidebar.image(st.session_state.img_original_display, caption="Imagen Original Reescalada", use_container_width=True) # AJUSTE 1

    st.subheader("Imagen Procesada")
    # Columna para RGB y Columna para el feedback binario (AJUSTE 2)
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.img_original_display, caption="RGB", use_container_width=True) # AJUSTE 1
    
    st.sidebar.subheader("Parámetros de Segmentación Watershed")
    # AJUSTE 3: Valores por defecto de los sliders
    threshold_min = st.sidebar.slider("Umbral Mínimo", 0, 254, 150, key="thresh_min")
    threshold_max = st.sidebar.slider("Umbral Máximo", 1, 255, 255, key="thresh_max")
    kernel_size = st.sidebar.slider("Tamaño del Kernel", 1, 20, 3, step=2, key="kernel_size")
    iterations = st.sidebar.slider("Iteraciones", 0, 20, 8, key="iterations")
    padding_cells = st.sidebar.slider("Padding (Células)", 0, 100, 10, key="padding_cells")
    
    # --- SECCIÓN: FEEDBACK BINARIO INTERACTIVO ---
    # AJUSTE 2: Ahora se muestra al lado de la imagen RGB
    if st.session_state.gray_img_display is not None:
        with col2: # Mostrar el feedback binario en la segunda columna (junto a RGB)
            st.image(
                get_binary_feedback_image(
                    st.session_state.gray_img_display,
                    threshold_min,
                    threshold_max,
                    kernel_size,
                    iterations
                ),
                caption="Feedback Binario",
                use_container_width=True, # AJUSTE 1
                clamp=True
            )
    # --- FIN SECCIÓN ---


    if st.sidebar.button("Ejecutar Análisis", key="run_analysis_button"):
        if loaded_model_h5 is None or loaded_tissue_model is None:
            st.error("Uno o ambos modelos no pudieron cargarse. No se puede ejecutar el análisis.")
        elif st.session_state.img_original_display is None:
            st.error("No hay ninguna imagen cargada para analizar.")
        else:
            st.cache_data.clear()
            
            st.subheader("Paso 1: Segmentación de Células")
            segments_rgb_all, segment_ids_processed, markers = apply_watershed_and_extract_cells(
                st.session_state.img_original_display, st.session_state.gray_img_display, 
                threshold_max, threshold_min, kernel_size, iterations, padding_cells
            )
            st.session_state.segments_rgb_all = segments_rgb_all
            st.session_state.segment_ids_processed = segment_ids_processed
            st.session_state.watershed_markers = markers
            
            fig_markers = plt.figure(figsize=(10, 5))
            ax1 = fig_markers.add_subplot(1, 2, 1)
            ax1.imshow(st.session_state.img_original_display)
            ax1.set_title('Imagen Original Reescalada')
            ax1.axis('off')

            ax2 = fig_markers.add_subplot(1, 2, 2)
            ax2.imshow(st.session_state.watershed_markers, cmap='nipy_spectral')
            ax2.set_title('Marcadores de Segmentos (Watershed)')
            ax2.axis('off')
            st.pyplot(fig_markers)


            st.subheader("Paso 2: Extracción de Segmentos de Tejido")
            grid_segments_224x224_array = extract_tissue_patches(st.session_state.img_original_display)
            st.session_state.grid_segments_224x224_array = grid_segments_224x224_array

            st.subheader("Paso 3: Clasificación y Decisión Compuesta")
            average_probabilities, predicted_classes_cells_individual, cells_for_prediction_normalized = classify_cells_and_average(
                st.session_state.segments_rgb_all, loaded_model_h5, class_labels
            )
            st.session_state.cells_for_prediction_normalized = cells_for_prediction_normalized
            st.session_state.predicted_classes_cells_individual = predicted_classes_cells_individual


            average_tissue_probabilities, predicted_classes_tissue_individual, tissue_for_prediction_normalized = classify_tissue_and_average(
                st.session_state.grid_segments_224x224_array, loaded_tissue_model, class_labels
            )
            st.session_state.tissue_for_prediction_normalized = tissue_for_prediction_normalized
            st.session_state.predicted_classes_tissue_individual = predicted_classes_tissue_individual

            st.session_state.combined_average_probabilities = (average_probabilities + average_tissue_probabilities) / 2.0
            st.session_state.overall_combined_predicted_class = np.argmax(st.session_state.combined_average_probabilities)

            st.write("### Resultados Globales")
            st.write(f"**Probabilidades Promedio (Modelo Células):** {average_probabilities}")
            st.write(f"**Probabilidades Promedio (Modelo Tejido):** {average_tissue_probabilities}")
            st.write(f"**Probabilidades Promedio Combinadas:** {st.session_state.combined_average_probabilities}")
            st.markdown(f"**Decisión Final Compuesta:** **<span style='color:green;'>{class_labels[st.session_state.overall_combined_predicted_class]}</span>**", unsafe_allow_html=True)
            
            fig_combined_probs = plt.figure(figsize=(6, 4))
            plt.bar(class_labels, st.session_state.combined_average_probabilities, color=['skyblue', 'lightcoral'][:num_classes])
            plt.ylim(0, 1)
            plt.title('Probabilidades Combinadas Globales')
            plt.ylabel('Probabilidad Promedio')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig_combined_probs)

            st.session_state.show_saliency = True
    else:
        st.session_state.show_saliency = False

    if st.session_state.show_saliency:
        st.subheader("Paso 4: Mapas de Saliencia (Ejemplos Aleatorios)")

        st.write("#### Células")
        num_saliency_examples = 5
        if st.session_state.segments_rgb_all.shape[0] > 0:
            selected_cell_indices = random.sample(range(st.session_state.segments_rgb_all.shape[0]), min(num_saliency_examples, st.session_state.segments_rgb_all.shape[0]))
            
            fig_cell_saliency, axes_cell_saliency = plt.subplots(len(selected_cell_indices), 2, figsize=(8, len(selected_cell_indices) * 2.5))
            fig_cell_saliency.suptitle('Saliencia en Segmentos de Células', fontsize=14, y=1.02)
            
            if len(selected_cell_indices) == 1: axes_cell_saliency = np.expand_dims(axes_cell_saliency, axis=0)

            for i, idx in enumerate(selected_cell_indices):
                cell_img_original = st.session_state.segments_rgb_all[idx]
                cell_img_for_pred_normalized = st.session_state.cells_for_prediction_normalized[idx]
                predicted_class_idx = st.session_state.predicted_classes_cells_individual[idx]

                saliency_map = compute_saliency_map(loaded_model_h5, cell_img_for_pred_normalized, predicted_class_idx)
                saliency_map_resized_to_original = cv.resize(saliency_map, (cell_img_original.shape[1], cell_img_original.shape[0]), interpolation=cv.INTER_LINEAR)
                # AJUSTE 1 (alpha) y AJUSTE 4 (cmap, boost, kernel)
                overlayed_cell_img = overlay_saliency_map(
                    cell_img_original, saliency_map_resized_to_original, alpha=0.3, # Alpha más bajo para más transparencia
                    cmap='jet', saliency_boost_factor=1.2, # Ajustes para el efecto relieve
                    smooth_saliency_kernel=(3,3) # Kernel más grande para mayor suavidad
                )
                
                axes_cell_saliency[i, 0].imshow(cell_img_original)
                axes_cell_saliency[i, 0].set_title(f"Original (ID:{idx})\nPred: {class_labels[predicted_class_idx]}")
                axes_cell_saliency[i, 0].axis('off')
                axes_cell_saliency[i, 1].imshow(overlayed_cell_img)
                axes_cell_saliency[i, 1].set_title(f"Saliencia (ID:{idx})")
                axes_cell_saliency[i, 1].axis('off')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            st.pyplot(fig_cell_saliency)
        else:
            st.write("No hay segmentos de células para mostrar mapas de saliencia.")

        st.write("#### Tejido")
        if st.session_state.grid_segments_224x224_array.shape[0] > 0:
            selected_tissue_indices = random.sample(range(st.session_state.grid_segments_224x224_array.shape[0]), min(num_saliency_examples, st.session_state.grid_segments_224x224_array.shape[0]))
            
            fig_tissue_saliency, axes_tissue_saliency = plt.subplots(len(selected_tissue_indices), 2, figsize=(8, len(selected_tissue_indices) * 2.5))
            fig_tissue_saliency.suptitle('Saliencia en Segmentos de Tejido', fontsize=14, y=1.02)

            if len(selected_tissue_indices) == 1: axes_tissue_saliency = np.expand_dims(axes_tissue_saliency, axis=0)

            for i, idx in enumerate(selected_tissue_indices):
                tissue_img_original = st.session_state.grid_segments_224x224_array[idx]
                tissue_img_for_pred_normalized = st.session_state.tissue_for_prediction_normalized[idx]
                predicted_class_idx = st.session_state.predicted_classes_tissue_individual[idx]

                saliency_map = compute_saliency_map(loaded_tissue_model, tissue_img_for_pred_normalized, predicted_class_idx)
                # AJUSTE 1 (alpha) y AJUSTE 4 (cmap, boost, kernel)
                overlayed_tissue_img = overlay_saliency_map(
                    tissue_img_original, saliency_map, alpha=0.3, # Alpha más bajo para más transparencia
                    cmap='jet', saliency_boost_factor=1.2, # Ajustes para el efecto relieve
                    smooth_saliency_kernel=(3,3) # Kernel más grande para mayor suavidad
                )
                
                axes_tissue_saliency[i, 0].imshow(tissue_img_original)
                axes_tissue_saliency[i, 0].set_title(f"Original (ID:{idx})\nPred: {class_labels[predicted_class_idx]}")
                axes_tissue_saliency[i, 1].imshow(overlayed_tissue_img)
                axes_tissue_saliency[i, 1].set_title(f"Saliencia (ID:{idx})")
                axes_tissue_saliency[i, 0].axis('off')
                axes_tissue_saliency[i, 1].axis('off')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            st.pyplot(fig_tissue_saliency)
        else:
            st.write("No hay segmentos de tejido para mostrar mapas de saliencia.")
else:
    st.info("Por favor, carga una imagen para comenzar el análisis.")