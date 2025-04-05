import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

@tf.keras.utils.register_keras_serializable()
class CapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = tf.keras.initializers.get('glorot_uniform')

    def build(self, input_shape):
        assert len(input_shape) >= 3, "Input Tensor shape should be [None, input_num_capsule, input_dim_capsule]"

        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transformation matrix
        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule,
                                        self.input_dim_capsule, self.dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W',
                                 trainable=True)
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        # Expand dims to perform batch matrix multiplication
        inputs_expand = tf.expand_dims(inputs, 2)
        inputs_tiled = tf.expand_dims(inputs_expand, 3)

        # Compute "prediction vectors" by applying transformation matrix W
        u_hat = tf.matmul(inputs_tiled, self.W)  # shape: [batch_size, input_caps, num_caps, 1, dim_caps]
        u_hat = tf.squeeze(u_hat, axis=-2)

        # Routing algorithm
        b = tf.zeros(shape=[tf.shape(inputs)[0], self.input_num_capsule, self.num_capsule])

        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            c = tf.expand_dims(c, axis=-1)
            s = tf.reduce_sum(c * u_hat, axis=1)  # weighted sum
            v = self.squash(s)

            if i < self.routings - 1:
                v_expand = tf.expand_dims(v, axis=1)
                b += tf.reduce_sum(u_hat * v_expand, axis=-1)

        return v

    def squash(self, s, axis=-1):
        s_norm = tf.norm(s, axis=axis, keepdims=True)
        scale = (s_norm**2) / (1 + s_norm**2) / (s_norm + tf.keras.backend.epsilon())
        return scale * s

    def get_config(self):
        config = super(CapsuleLayer, self).get_config()
        config.update({
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
