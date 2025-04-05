import tensorflow as tf
import numpy as np
import gdown
import os

GDRIVE_URL = "https://drive.google.com/uc?id=1ABM_i0WrNQf1OBw8fePznJNczWUNE2LR"  # Replace with your actual file ID
MODEL_PATH = "./pcos_capsnet_eval_model.h5"

def load_model_from_gdrive():
    if not os.path.exists(MODEL_PATH):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

def predict(model, image_array):
    """
    Predicts the label for a single image.
    Returns: (label, confidence percentage)
    """
    y_pred, _ = model.predict(image_array)
    confidence = np.max(y_pred)
    label = "Infected (PCOS)" if np.argmax(y_pred) == 1 else "Healthy"
    return label, confidence * 100
