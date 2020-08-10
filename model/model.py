import keras
import tensorflow as tf

from keras.layers import *
from model.layers import *
from pretrain_config import *


class Bert(object):
    def __init__(self,
                 vocab_size,  # 词表大小
                 hidden_size,  # 编码维度
                 num_hidden_layers,  # Transformer总层数
                 num_attention_heads,  # Attention的头数
                 intermediate_size,  # FeedForward的隐层维度
                 hidden_act,  # FeedForward隐层的激活函数
                 max_position_embeddings,  # 最大序列长度
                 hidden_dropout_prob=None,  # Dropout比例
                 embedding_size=None,  # 是否指定embedding_size
                 attention_key_size=None,  # Attention中Q,K的head_size
                 sequence_length=None,  # 是否固定序列长度
                 keep_tokens=None,  # 要保留的词ID列表
                 auxiliary_embeddings=None,  # 要增加的embeddings
                 # layers=None,  # 外部传入的Keras层
                 prefix=None,  # 层名前缀
                 name=None,  # 模型名称
                 **kwargs):
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if auxiliary_embeddings is not None:
            vocab_size += len(auxiliary_embeddings)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout_rate = hidden_dropout_prob or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.auxiliary_embeddings = auxiliary_embeddings
        self.attention_mask = None
        self.position_bias = None
        # self.layers = {} if layers is None else layers
        self.prefix = prefix or ''
        self.name = name
        self.built = False

    # def call(self):
    #     pass

    def build(self, additional_input_layers):
        # 根据生成bert数据的部分可知：BERT的输入是token_ids和segment_ids
        # 构建输入层
        token_in = Input(shape=(self.sequence_length, ), name='Input-Token')
        segment_in = Input(shape=(self.sequence_length, ), name='Input-Segment')
        if additional_input_layers:
            if isinstance(additional_input_layers, list):
                self.inputs = [token_in, segment_in].extend(additional_input_layers)
            else:
                self.inputs = [token_in, segment_in, additional_input_layers]

        # 构建bert的embedding层
        x = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size, mask_zero=True,
                      embeddings_initializer=keras.initializers.truncated_normal(stddev=0.02),
                      name='Embedding-Token')(token_in)
        s = Embedding(input_dim=2, output_dim=self.embedding_size, mask_zero=True,
                      embeddings_initializer=keras.initializers.truncated_normal(stddev=0.02),
                      name='Embedding-Segment')(segment_in)

        # 加入类型信息
        x = Add()([x, s])

        # 加入位置信息
        x = PositionEmbedding(input_dim=self.sequence_length, output_dim=self.hidden_size,
                              embeddings_initializer=keras.initializers.zeros,
                              name='Embedding-Position')(x)


















