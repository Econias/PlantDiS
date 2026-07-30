import os
import numpy as np
import streamlit as st
from PIL import Image

# Use lightweight tflite_runtime if available, otherwise fall back to full tf
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# --- Page Setup & Dark Theme CSS ---
st.set_page_config(page_title='Plant Disease Detector', layout='centered')

st.markdown("""
    <style>
    /* Dark background styling */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Header title formatting */
    h1 {
        color: #00E676 !important;
        font-family: 'Inter', sans-serif;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* Card container for upload/results */
    div[data-testid="stFileUploader"], div.stButton {
        display: flex;
        justify-content: center;
    }
    
    /* Custom primary action button */
    .stButton > button {
        background-color: #00C853 !important;
        color: #FFFFFF !important;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #00E676 !important;
        box-shadow: 0 4px 15px rgba(0, 230, 118, 0.4);
    }
    
    /* Prediction output card */
    .result-card {
        background-color: #1E2631;
        border-left: 5px solid #00E676;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-title {
        color: #00E676;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .result-value {
        color: #FFFFFF;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title('🌿 Plant Disease Detector')

# --- 1. Load the TFLite Model ---
@st.cache_resource
def load_tflite_model():
    model_path = 'plantvillage_efficientnet_b0.tflite'
    if not os.path.exists(model_path):
        st.error(f'Model file `{model_path}` not found in repository root!')
        return None, None
    
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

classes = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot',
    'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# --- 2. Image Preprocessing for EfficientNet ---
def preprocess_image(image):
    # Resize to EfficientNet input dimension (224x224)
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)

    # EfficientNet-B0 expectation: scale values to [0, 255] float
    # If your TFLite model was quantized to INT8, cast to np.uint8 instead.
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- 3. UI Flow ---
uploaded_file = st.file_uploader('Choose a leaf image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Predict Disease'):
        if interpreter is None:
            st.error("Model interpreter failed to load.")
        else:
            with st.spinner('Analyzing leaf features...'):
                # Prepare TFLite input/output details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Preprocess input image array
                input_data = preprocess_image(image)

                # Set input tensor and invoke inference
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                # Extract predictions
                preds = interpreter.get_tensor(output_details[0]['index'])[0]
                
                # Apply Softmax if model output raw logits
                if np.max(preds) > 1.0 or np.min(preds) < 0.0:
                    preds = np.exp(preds) / np.sum(np.exp(preds))

                top_idx = np.argmax(preds)
                confidence = preds[top_idx]

                # Render result card with HTML styling
                st.markdown(f"""
                    <div class="result-card">
                        <div class="result-title">Detection Result</div>
                        <div class="result-value"><b>Disease:</b> {classes[top_idx].replace('___', ' - ').replace('_', ' ')}</div>
                        <div class="result-value"><b>Confidence:</b> {confidence:.2%}</div>
                    </div>
                """, unsafe_allow_html=True)
