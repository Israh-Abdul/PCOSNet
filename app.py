import streamlit as st
from model_utils import load_model_from_gdrive, predict
from preprocessing import preprocess_image
from PIL import Image
import numpy as np

# Set Streamlit page config
st.set_page_config(page_title="PCOS Detection App", layout="centered")

st.title("🧬 PCOS Detection from Ultrasound Image")
st.write("Upload an ultrasound image to predict whether it shows signs of PCOS.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load model from Google Drive (only once)
@st.cache_resource
def load_model():
    return load_model_from_gdrive()

model = load_model()

# Predict
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(image, caption="Uploaded Ultrasound Image", use_column_width=True)

    if st.button("🔍 Predict"):
        processed_img = preprocess_image(image)
        label, confidence = predict(model, processed_img)

        st.subheader("🩺 Prediction:")
        st.write(f"**{label}** ({confidence:.2f}% confidence)")
