import keras
import keras.backend as K

from keras.layers import Layer
from keras import initializers


class PositionEmbedding(Layer):
    def __init__(
            self,
            input_dim,
            output_dim,
            embeddings_initializer=keras.initializers.zeros,
            **kwargs
    ):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embedding = self.add_weight(name='position_embeddings',
                                         shape=(self.input_dim, self.output),
                                         initializer=self.embeddings_initializer)
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        pos_embeddings = self.embedding[:seq_len]
        pos_embeddings = K.expand_dims(pos_embeddings, 0)
        return inputs + pos_embeddings

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embeddings_initializer': initializers.serialize(self.embeddings_initializer)
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))
