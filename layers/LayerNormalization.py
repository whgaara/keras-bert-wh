import keras
import keras.backend as K

from keras.layers import Layer
from keras import initializers


class LayerNormalization(Layer):
    def __init__(self, center=True, scale=True, epsilon=None, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.epsilon = epsilon or 1e-12

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)
        shape = input_shape[-1]
        if self.center:
            self.beta = self.add_weight(shape=shape, initializer='zeros', name='beta')
        if self.scale:
            self.gamma = self.add_weight(shape=shape, initializer='ones', name='gamma')

    def call(self, inputs, **kwargs):
        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std
            outputs = outputs * self.gamma
        if self.center:
            outputs = outputs + self.beta
        return outputs

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
