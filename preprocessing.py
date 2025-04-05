import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

def preprocess_image(image, target_size=(128, 128)):
    """
    Convert PIL image to model-compatible array.
    """
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
