import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title='Plant Disease Detector', layout='centered')
st.title('🌿 Plant Disease Detector')

# 1. Load the model
@st.cache_resource
def load_my_model():
    model_path = 'plantvillage_efficientnet_b0.keras'
    if not os.path.exists(model_path):
        st.error(f'Model file {model_path} not found!')
        return None
    return tf.keras.models.load_model(model_path, compile=False)

model = load_my_model()

classes = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot',
    'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

uploaded_file = st.file_uploader('Choose a leaf image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Predict'):
        with st.spinner('Analyzing...'):
            # Preprocess
            img = np.array(image)
            img = tf.image.resize(img, (224, 224))
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            img = tf.expand_dims(img, axis=0)

            # Predict
            preds = model.predict(img)[0]
            top_idx = np.argmax(preds)
            
            st.success(f'**Result: {classes[top_idx]}**')
            st.write(f'Confidence: {preds[top_idx]:.2%}')