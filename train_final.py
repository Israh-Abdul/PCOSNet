import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Define constants
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 1  # Binary classification

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
train_generator = train_datagen.flow_from_directory(
    'C:/Users/fayiz/OneDrive/Desktop/PROJECT/SIMPLECAPSNET/dataset/',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'C:/Users/fayiz/OneDrive/Desktop/PROJECT/SIMPLECAPSNET/dataset/',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Define Capsule Network Layer
def squash(x, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(x), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + 1e-8)
    return scale * x

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsules, routing_iters=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routing_iters = routing_iters

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[1], self.num_capsules, self.dim_capsules, input_shape[2]),
                                initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, -1)
        u_hat = tf.linalg.matmul(self.W, inputs, transpose_a=True)
        u_hat = tf.transpose(u_hat, perm=[0, 2, 3, 1])
        
        b = tf.zeros(shape=(tf.shape(inputs)[0], self.num_capsules, tf.shape(inputs)[1]))
        for _ in range(self.routing_iters):
            c = tf.nn.softmax(b, axis=1)
            s = tf.reduce_sum(c * u_hat, axis=2)
            v = squash(s)
            if _ < self.routing_iters - 1:
                b += tf.reduce_sum(u_hat * tf.expand_dims(v, 2), axis=-1)
        return v

# Define CapsNet model
def build_capsnet_model():
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = layers.Conv2D(256, (9, 9), activation='relu', padding='valid')(inputs)
    x = layers.Reshape(target_shape=[-1, 8])(x)
    x = CapsuleLayer(num_capsules=8, dim_capsules=16)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)

    model = models.Model(inputs, x)
    return model

# Build and compile the model
model = build_capsnet_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

# Save the model
model_save_path = 'C:/Users/fayiz/OneDrive/Desktop/PROJECT/SIMPLECAPSNET/capsnet.h5'
model.save(model_save_path)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy*100:.2f}%")

print("Training completed")

# Save training history
history_path = 'history.pckl'
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)

# Load and plot training history
with open(history_path, 'rb') as f:
    history = pickle.load(f)

plt.figure()
plt.plot(history.get('accuracy', []), label='Training Accuracy')
plt.plot(history.get('val_accuracy', []), label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
