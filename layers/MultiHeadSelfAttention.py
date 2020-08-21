import math
import keras
import tensorflow as tf
import keras.backend as K

from keras.layers import Layer, Dense, initializers


class MultiHeadSelfAttention(Layer):
    def __init__(self,
                 attention_head_num,
                 attention_head_size,
                 kernel_initializer,
                 attention_mask=None,
                 use_bias=True,
                 size_per_head=512,
                 **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.attention_head_num = attention_head_num
        self.attention_head_size = attention_head_size
        self.out_dim = attention_head_num * attention_head_size
        self.kernel_initializer = kernel_initializer
        self.attention_mask = attention_mask
        self.use_bias = use_bias
        self.size_per_head = size_per_head

    def build(self, input_shape):
        super(MultiHeadSelfAttention, self).build(input_shape)
        self.q_layer = Dense(units=self.out_dim,
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer,
                             name='MultiHeadSelfAttention-Q')
        self.k_layer = Dense(units=self.out_dim,
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer,
                             name='MultiHeadSelfAttention-K')
        self.v_layer = Dense(units=self.out_dim,
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer,
                             name='MultiHeadSelfAttention-V')
        self.o_layer = Dense(units=self.out_dim,
                             use_bias=self.use_bias,
                             kernel_initializer=self.kernel_initializer,
                             name='MultiHeadSelfAttention-Out')

    def call(self, inputs, **kwargs):
        # 对q、k、v分别进行计算并reshape
        qx, kx, vx = inputs
        q = self.q_layer(qx)
        k = self.k_layer(kx)
        v = self.v_layer(vx)

        # 先将batch_size*seq_len*embedding_size变成batch_size*seq_len*head*head_size
        # 再将batch_size*seq_len*head*head_size转置成batch_size*head*seq_len*head_size
        q = K.reshape(q, (-1, K.shape(qx)[1], self.attention_head_num, self.attention_head_size))
        q = tf.transpose(q, [0, 2, 1, 3])
        k = K.reshape(k, (-1, K.shape(kx)[1], self.attention_head_num, self.attention_head_size))
        k = tf.transpose(k, [0, 2, 1, 3])
        v = K.reshape(v, (-1, K.shape(vx)[1], self.attention_head_num, self.attention_head_size))
        v = tf.transpose(v, [0, 2, 1, 3])
        # q与k的转置相乘得到：[batch_size, head, seq_len, seq_len]
        attention_scores = K.batch_dot(q, k, axes=[3, 3])
        # 因为q、k相乘，结果变大，因此对结果除以根号512
        # attention_scores = keras.layers.multiply([attention_scores, 1.0 / math.sqrt(float(self.size_per_head))])
        attention_scores = attention_scores / math.sqrt(float(self.size_per_head))

        # 防止padding补全的0经过softmax后影响结果，对每个0值都加一个很大的负数，这样softmax后也会约等于0
        # attention_mask的shape为：[batch_size, seq_len, seq_len]
        if self.attention_mask is not None:
            # [batch_size, 1, seq_len, seq_len]
            self.attention_mask = K.expand_dims(self.attention_mask, 1)
            add_mask = (1.0 - K.cast(self.attention_mask, K.floatx())) * 1e5
            attention_scores -= add_mask

        attention_probs = K.softmax(attention_scores)
        attention_probs = K.batch_dot(attention_probs, v)
        attention_probs = K.reshape(attention_probs, [-1, K.shape(qx)[1], self.out_dim])
        attention_probs = self.o_layer(attention_probs)
        return attention_probs

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = {
            'attention_head_num': self.attention_head_num,
            'attention_head_size': self.attention_head_size,
            'out_dim': self.out_dim,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'attention_mask': self.attention_mask,
            'use_bias': self.use_bias,
            'size_per_head': self.size_per_head
        }
        base_config = super(MultiHeadSelfAttention, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))
