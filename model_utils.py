import tensorflow as tf
import numpy as np
import gdown
import os
from capsule_layers import CapsuleLayer, Length, Mask

GDRIVE_URL = "https://drive.google.com/uc?id=1ABM_i0WrNQf1OBw8fePznJNczWUNE2LR"  # Replace with your actual file ID
MODEL_PATH = "./pcos_capsnet_eval_model.h5"

@tf.keras.utils.register_keras_serializable()
def custom_objects():
    return {
        "CapsuleLayer": CapsuleLayer,
        "Length": Length,
        "Mask": Mask,
    }

@st.cache_resource
def load_model_from_gdrive():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    
    print("Loading model...")
    return tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects(), compile=False)

def predict(model, preprocessed_image):
    prediction = model.predict(tf.expand_dims(preprocessed_image, axis=0))
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
    confidence = tf.reduce_max(prediction).numpy()
    return predicted_class, confidence
