import keras

from keras.layers import Layer


class AddLayer(Layer):
    def build(self, input_shape):
        super(AddLayer, self).build(input_shape)
        out_dim = input_shape[-1]
        self.bias = self.add_weight(
            name='add-bias',
            shape=(out_dim,),
            initializer='zero',
            trainable=True
        )

    def call(self, inputs, **kwargs):
        return keras.backend.bias_add(inputs, self.bias)

    def compute_mask(self, inputs, mask=None):
        return mask
