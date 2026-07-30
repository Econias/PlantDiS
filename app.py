import streamlit as st
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for styling cards, shadows, and headers
st.markdown("""
    <style>
    /* Clean background & card padding */
    .stApp {
        background-color: #f9fbf9;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    /* Custom header design */
    .title-text {
        color: #1e3a1e;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/color/96/leaf.png", width=64)
    st.title("PlantGuard AI")
    st.caption("EfficientNet-B0 Disease Classifier")
    st.divider()
    st.markdown("### How it works")
    st.markdown("""
    1. Upload a clear photo of an affected leaf.
    2. The model analyzes leaf textures and spots.
    3. View instant diagnosis and care tips.
    """)
    st.divider()
    st.info("Supported crops: Pepper, Potato, Tomato, and more.")

# 4. Main Page Header
st.title("🌿 Leaf Disease Analyzer")
st.write("Upload an image of a plant leaf to detect potential diseases instantly.")
st.divider()

# 5. Two-Column Dashboard Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📷 Upload Sample")
    uploaded_file = st.file_uploader(
        "Choose a leaf image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image under good lighting"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("🔍 Analysis & Diagnosis")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing leaf features..."):
            # --- Place your model inference code here ---
            # E.g., processed_img = preprocess(image)
            # predictions = interpreter.predict(...)
            
            # Simulated dummy outputs for UI preview:
            top_class = "Potato___Early_blight"  
            confidence = 0.942
            # ---------------------------------------------

        # Display Top Match Badge
        is_healthy = "healthy" in top_class.lower()
        badge_color = "🟢" if is_healthy else "🔴"
        formatted_name = top_class.replace("__", " - ").replace("_", " ")

        st.markdown(f"### {badge_color} **{formatted_name}**")
        
        # Confidence Gauge
        st.write(f"**Model Confidence:** {confidence * 100:.1f}%")
        st.progress(float(confidence))

        st.divider()

        # Actionable Advice Accordion
        if not is_healthy:
            with st.expander("🛠️ Recommended Action & Care Tips", expanded=True):
                st.markdown("""
                * **Isolation:** Separate affected plant to prevent airborne spore propagation.
                * **Treatment:** Apply copper-based fungicide or bio-fungicide weekly.
                * **Pruning:** Remove and safely discard visibly damaged foliage.
                * **Watering:** Water at the base of the plant to keep leaf surface dry.
                """)
        else:
            st.success("The leaf appears healthy! Keep maintaining current crop care routines.")

    else:
        st.info("👆 Upload an image on the left to start the diagnosis.")
