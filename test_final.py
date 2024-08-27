import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os


# Define constants
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# Load the trained model
model = tf.keras.models.load_model('trancnn.h5')

# Data Augmentation for testing
test_datagen = ImageDataGenerator(rescale=1./255)

# Load test data
test_generator = test_datagen.flow_from_directory(
    'C:/Users/fayiz/OneDrive/Desktop/PROJECT/SIMPLECNN/dataset',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Evaluate the model on the test data
#loss, accuracy = model.evaluate(test_generator)
#print(f"Test Accuracy: {accuracy*100:.2f}%")

# Predict on new images
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)
    class_idx = int(np.round(prediction[0][0]))
    classes = list(test_generator.class_indices.keys())
    
    print(f"Predicted class for {image_path}: {classes[class_idx]}")
    print("Maximum Probability: ",np.max(prediction[0], axis=-1))

# Example prediction
example_image_path = 'C:/Users/fayiz/OneDrive/Desktop/PROJECT/SIMPLECNN/dataset/test/test1.jpeg'  # replace with your image path
predict_image(example_image_path)

#C:/Users/fayiz/OneDrive/Desktop/PROJECT/SIMPLECNN/dataset/test/test1.jpeg



