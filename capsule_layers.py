import tensorflow as tf
from tensorflow.keras import layers, initializers, backend as K

def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_initializer='glorot_uniform', **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')
        super().build(input_shape)

    def call(self, inputs, training=None):
        inputs_expand = tf.expand_dims(inputs, 1)
        inputs_tiled  = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_tiled  = tf.expand_dims(inputs_tiled, 4)
        inputs_hat = tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled)

        b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule, 1, 1])

        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = tf.reduce_sum(c * inputs_hat, axis=2, keepdims=True)
            outputs = squash(outputs, axis=-2)
            if i < self.routings - 1:
                outputs_tiled = tf.tile(outputs, [1, 1, self.input_num_capsule, 1, 1])
                agreement = tf.matmul(inputs_hat, outputs_tiled, transpose_a=True)
                b += agreement

        outputs = tf.squeeze(outputs, [2, 4])
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    def get_config(self):
        config = super(CapsuleLayer, self).get_config()
        config.update({
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel_size,
                           strides=strides, padding=padding, name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)
