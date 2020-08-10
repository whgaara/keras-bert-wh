import keras
import keras.backend as K

from keras.layers import Layer


class PositionEmbedding(Layer):
    def __init__(
            self,
            input_dim,
            output_dim,
            embeddings_initializer=keras.initializers.zeros,
            **kwargs
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding = None
        self.embeddings_initializer = embeddings_initializer

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embedding = self.add_weight(name='position_embeddings',
                                         shape=(self.input_dim, self.output),
                                         initializer=self.embeddings_initializer)

    def call(self, inputs, **kwargs):
        input_shape = K.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        pos_embeddings = self.embedding[:seq_len]
        pos_embeddings = K.expand_dims(pos_embeddings, 0)
        return inputs + pos_embeddings











