from keras.layers import Layer, Dense


class MultiHeadSelfAttention(Layer):
    def __init__(self,
                 attention_head_num,
                 attention_head_size,
                 kernel_initializer,
                 attention_scale=True,
                 use_bias=True,
                 **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.attention_head_num = attention_head_num
        self.attention_head_size = attention_head_size
        self.out_dim = attention_head_num * attention_head_size
        self.kernel_initializer = kernel_initializer
        self.attention_scale=attention_scale
        self.use_bias = use_bias

    def build(self, input_shape):
        super(MultiHeadSelfAttention, self).build(input_shape)
        q = Dense(units=self.out_dim,
                  use_bias=self.use_bias,
                  kernel_initializer=self.kernel_initializer,
                  name='MultiHeadSelfAttention-Q')
        k = Dense(units=self.out_dim,
                  use_bias=self.use_bias,
                  kernel_initializer=self.kernel_initializer,
                  name='MultiHeadSelfAttention-K')
        v = Dense(units=self.out_dim,
                  use_bias=self.use_bias,
                  kernel_initializer=self.kernel_initializer,
                  name='MultiHeadSelfAttention-V')
        o = Dense(units=self.out_dim,
                  use_bias=self.use_bias,
                  kernel_initializer=self.kernel_initializer,
                  name='MultiHeadSelfAttention-Out')

    def call(self, inputs, **kwargs):
        pass

    def get_config(self):
        pass
