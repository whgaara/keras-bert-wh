import keras

from keras.layers import *
from keras.models import Model
from layers.PositionEmbedding import PositionEmbedding
from layers.LayerNormalization import LayerNormalization
from layers.MultiHeadSelfAttention import MultiHeadSelfAttention
from layers.FeedForward import FeedForward
from layers.AddLayer import AddLayer


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
        self.embeddings_x = Embedding(input_dim=self.vocab_size, output_dim=self.hidden_size, mask_zero=True,
                                      embeddings_initializer=initializers.truncated_normal(stddev=0.02),
                                      name='Embedding-Token')
        self.embeddings_s = Embedding(input_dim=2, output_dim=self.hidden_size,
                                      embeddings_initializer=initializers.truncated_normal(stddev=0.02),
                                      name='Embedding-Segment')

    def build(self):
        # 根据生成bert数据的部分可知：BERT的输入是token_ids和segment_ids
        token_in = Input(shape=(self.sequence_length,), name='Input-Token')
        segment_in = Input(shape=(self.sequence_length,), name='Input-Segment')
        self.inputs = [token_in, segment_in]
        self.outputs = self.call(self.inputs)
        if not isinstance(self.outputs, list):
            self.outputs = [self.outputs]
        self.model = Model(self.inputs, self.outputs, name=self.name)
        self.built = True

    def call(self, inputs):
        x, s = inputs[:2]

        # -----------------------------embedding层----------------------------- #
        x = self.embeddings_x(x)
        s = self.embeddings_s(s)
        # 加入类型信息
        x = Add(name='Embedding-Token-Segment')([x, s])
        # 加入位置信息: batch_size * sen_len * hidden_size
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
        attention_x = None
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
        # -----------------------------transformer层----------------------------- #

        # -----------------------------任务层：Pooler----------------------------- #
        pooler_x = Lambda(lambda x: x[:, 0], name='Pooler')(attention_x)
        pooler_x = Dense(units=self.hidden_size, activation='tanh', name='Pooler-Dense',
                         kernel_initializer=initializers.truncated_normal(stddev=0.02))(pooler_x)
        # -----------------------------任务层：Pooler----------------------------- #

        # -----------------------------任务层：mlm----------------------------- #
        mlm_x = Dense(units=self.hidden_size, activation=self.hidden_act,
                      kernel_initializer=initializers.truncated_normal(stddev=0.02), name='MLM-Dense')(attention_x)
        mlm_x = LayerNormalization(name='MLM-Norm')(mlm_x)
        embeddings_t = K.transpose(self.embeddings_x.embeddings)
        mlm_x = K.dot(mlm_x, embeddings_t)
        mlm_x = AddLayer(name='MLM-Bias')(mlm_x)
        outputs = Softmax(name='MLM-Activation')(mlm_x)
        # -----------------------------任务层：mlm----------------------------- #

        return attention_x, pooler_x, outputs
