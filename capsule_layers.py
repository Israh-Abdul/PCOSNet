import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):  # True label is provided
            inputs, mask = inputs
        else:  # If no true label, mask by max length
            x = K.sqrt(K.sum(K.square(inputs), -1))
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.shape[1])
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(
            shape=[self.input_num_capsule, self.num_capsule,
                   self.input_dim_capsule, self.dim_capsule],
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs, **kwargs):
        # Expand dims for matmul
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)
        W_expand = K.expand_dims(self.W, 0)
        u_hat = tf.matmul(inputs_expand, W_expand)
        u_hat = K.squeeze(u_hat, axis=-2)

        b = tf.zeros_like(u_hat[:, :, :, 0])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            s = tf.reduce_sum(tf.expand_dims(c, -1) * u_hat, axis=1)
            v = self.squash(s)
            if i < self.routings - 1:
                b += tf.reduce_sum(u_hat * tf.expand_dims(v, 1), axis=-1)
        return v

    def squash(self, s, axis=-1):
        s_squared_norm = K.sum(K.square(s), axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
        return scale * s

# Register for loading
@tf.keras.utils.register_keras_serializable()
class CapsuleLayer(CapsuleLayer): pass

@tf.keras.utils.register_keras_serializable()
class Length(Length): pass

@tf.keras.utils.register_keras_serializable()
class Mask(Mask): pass
