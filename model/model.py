import keras

from keras.layers import *
from keras.models import Model
from layers.PositionEmbedding import PositionEmbedding
from layers.LayerNormalization import LayerNormalization
from layers.MultiHeadSelfAttention import MultiHeadSelfAttention
from layers.FeedForward import FeedForward


class Bert(object):
    def __init__(self,
                 sequence_length,  # 句子的长度
                 vocab_size,  # 词表大小
                 hidden_size,  # 编码维度
                 num_hidden_layers,  # Transformer总层数
                 num_attention_heads,  # Attention的头数
                 intermediate_size,  # FeedForward的隐层维度
                 hidden_act,  # FeedForward隐层的激活函数
                 max_position_embeddings,  # 最大序列长度
                 hidden_dropout_prob=None,  # Dropout比例
                 name=None,  # 模型名称
                 ):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout_rate = hidden_dropout_prob or 0
        self.hidden_act = hidden_act
        self.name = name
        self.built = False

    def build(self):
        # 根据生成bert数据的部分可知：BERT的输入是token_ids和segment_ids
        token_in = Input(shape=(self.sequence_length,), name='Input-Token')
        segment_in = Input(shape=(self.sequence_length,), name='Input-Segment')
        inputs = [token_in, segment_in]
        outputs = self.call(inputs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        self.model = Model(inputs, outputs, name=self.name)

    def call(self, inputs):
        x, s = inputs[:2]

        # -----------------------------embedding层----------------------------- #
        x = Embedding(input_dim=self.vocab_size, output_dim=self.hidden_size, mask_zero=True,
                      embeddings_initializer=initializers.truncated_normal(stddev=0.02),
                      name='Embedding-Token')(x)
        s = Embedding(input_dim=2, output_dim=self.hidden_size,
                      embeddings_initializer=initializers.truncated_normal(stddev=0.02),
                      name='Embedding-Segment')(s)
        # 加入类型信息
        x = Add(name='Embedding-Token-Segment')([x, s])
        # 加入位置信息: batch_size * sen_len * embedding_size
        x = PositionEmbedding(input_dim=self.sequence_length, output_dim=self.hidden_size,
                              embeddings_initializer=initializers.zeros,
                              name='Embedding-Position')(x)
        # layer normalization ???
        x = LayerNormalization(name='Embedding-Norm')(x)
        # drop out
        x = Dropout(rate=self.dropout_rate, name='Embedding-Dropout')(x)
        # dense
        x = Dense(units=self.hidden_size, kernel_initializer=initializers.truncated_normal(stddev=0.02))(x)
        # -----------------------------embedding层----------------------------- #

        # -----------------------------transformer层----------------------------- #
        for index in range(self.num_hidden_layers):
            attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
            feed_forward_name = 'Transformer-%d-FeedForward' % index

            # MultiHeadSelfAttention
            new_x = [x, x, x]
            attention_x = MultiHeadSelfAttention(attention_head_num=self.num_attention_heads,
                                                 attention_head_size=self.attention_head_size,
                                                 kernel_initializer=initializers.truncated_normal(stddev=0.02),
                                                 name=attention_name)(new_x)

            # drop out
            attention_x = Dropout(rate=self.dropout_rate, name='%s-Dropout' % attention_name)(attention_x)

            # add
            attention_x = Add(name='%s-Add' % attention_name)([x, attention_x])

            # layer normalization
            attention_x = LayerNormalization(name='%s-Norm' % attention_name)(attention_x)

            # Feed Forward
            attention_x = FeedForward(
                units=self.intermediate_size,
                activation=self.hidden_act,
                use_bias=True,
                kernel_initializer=initializers.truncated_normal(stddev=0.02),
                name=feed_forward_name
            )(attention_x)

            # drop out
            attention_x = Dropout(rate=self.dropout_rate, name='%s-Dropout' % feed_forward_name)(attention_x)

            # add
            attention_x = Add(name='%s-Add' % feed_forward_name)([x, attention_x])

            # layer normalization
            attention_x = LayerNormalization(name='%s-Norm' % feed_forward_name)(attention_x)

            return attention_x
        # -----------------------------transformer层----------------------------- #

        # -----------------------------task层----------------------------- #

        # -----------------------------task层----------------------------- #



















