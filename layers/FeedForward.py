import keras
import tensorflow as tf
import keras.backend as K

from keras.layers import Layer, Dense, initializers


class FeedForward(Layer):
    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]
        self.dense1 = Dense(units=self.units,
                            activation=self.activation,
                            use_bias=self.use_bias,
                            kernel_initializer=self.kernel_initializer)
        self.dense2 = Dense(units=output_dim,
                            use_bias=self.use_bias,
                            kernel_initializer=self.kernel_initializer)

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))
