import keras.backend as K

from keras.layers import Layer


class TransposeDot(Layer):
    def __init__(self,
                 embeddings,
                 **kwargs):
        super(TransposeDot, self).__init__(**kwargs)
        self.embeddings = embeddings

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, **kwargs):
        self.mlm_x = inputs
        embeddings_t = K.transpose(self.embeddings)
        mlm_x = K.dot(self.mlm_x, embeddings_t)
        return mlm_x

    def compute_output_shape(self, input_shape):
        shape1 = K.int_shape(self.mlm_x)
        shape2 = K.int_shape(self.embeddings)
        batch_size = shape1[0]
        seq_length = shape1[1]
        out_dim = shape2[0]
        return (batch_size, seq_length, out_dim)
