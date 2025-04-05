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
        "Mask": Mask
    }
    
def download_model():
    print("Checking if model exists at:", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Creating directory and downloading...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        print("Download complete.")
    else:
        print("Model already exists locally.")

@tf.keras.utils.register_keras_serializable()
def load_model_from_gdrive():
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects(), compile=False)
    print("Model loaded successfully.")
    return model

def predict(model, image):
    prediction = model.predict(np.expand_dims(image, axis=0))[0][0]
    label = "PCOS Detected" if prediction > 0.5 else "Normal"
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    return label, confidence

