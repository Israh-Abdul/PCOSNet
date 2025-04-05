import tensorflow as tf
import numpy as np
import gdown
import os
from capsule_layers import CapsuleLayer

GDRIVE_URL = "https://drive.google.com/uc?id=1ABM_i0WrNQf1OBw8fePznJNczWUNE2LR"  # Replace with your actual file ID
MODEL_PATH = "./pcos_capsnet_eval_model.h5"

@tf.keras.utils.register_keras_serializable()

def download_model():
    """Download the model from Google Drive if it doesn't exist locally."""
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

def custom_objects():
    """Return a dictionary of custom objects for model loading."""
    return {'CapsuleLayer': CapsuleLayer}

def load_model_from_gdrive():
    """Download and load the trained CapsuleNet model."""
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects(), compile=False)
    return model

import numpy as np

def predict(model, image):
    # Expand dims for batch size and predict
    preds = model.predict(np.expand_dims(image, axis=0))  # shape: (1, 2, 16)
    
    # Calculate the length of each capsule vector (i.e., probability of each class)
    probs = np.linalg.norm(preds, axis=-1)  # shape: (1, 2)
    predicted_class = np.argmax(probs)
    
    label = "PCOS Detected" if predicted_class == 1 else "Normal"
    confidence = probs[0][predicted_class] * 100
    return label, confidence

