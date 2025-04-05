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

import os
import gdown
import tensorflow as tf
from capsule_layers import CapsuleLayer, Length, Mask

MODEL_PATH = "./models/pcos_capsnet_eval_model.h5"
GDRIVE_URL = "https://drive.google.com/uc?id=YOUR_MODEL_FILE_ID"  # replace with actual ID

def custom_objects():
    return {
        "CapsuleLayer": CapsuleLayer,
        "Length": Length,
        "Mask": Mask
    }

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Downloading model from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

@tf.keras.utils.register_keras_serializable()
def load_model_from_gdrive():
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects(), compile=False)
    return model
