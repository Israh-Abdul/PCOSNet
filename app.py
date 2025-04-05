import streamlit as st
from PIL import Image
import numpy as np
from model_utils import load_model_from_gdrive, predict

st.set_page_config(page_title="PCOS Detector", layout="centered")
st.title("🧠 PCOS Detection from Ultrasound Images")
st.markdown("Upload an ultrasound image to detect **PCOS** using a Capsule Network model.")

# --- Load the CapsuleNet model (with custom layers) from Google Drive ---
@st.cache_resource
def load_model():
    return load_model_from_gdrive()

model = load_model()

# --- Preprocessing function ---
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=-1)  # Shape: (128, 128, 1)
    return image_array

# --- File uploader UI ---
uploaded_file = st.file_uploader("Upload an ultrasound image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    processed_image = preprocess_image(image)
    label, confidence = predict(model, processed_image)

    if label == "PCOS Detected":
        st.error(f"⚠️ {label} with {confidence:.2f}% confidence.")
    else:
        st.success(f"✅ {label}. Confidence: {confidence:.2f}%")

st.markdown("---")
st.caption("Built with ❤️ using Capsule Networks and Streamlit")
