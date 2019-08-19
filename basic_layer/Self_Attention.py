'''
    Self_Attention(size_per_head, embedding_len, nb_head)
        input:
            size_per_head: 输出维度
            embedding_len： 隐藏层的维度（x转换成a的维度） 可以任意指定
            nb_head: multi_head 的层数
        output:
            Self_Attention 对象

    __call__(inputs):
        input: 
            输入的数据  [batch_size, ..., sequence_len, embedding_len]
        output:
            返回的结果  [batch_size, ..., sequence_len, size_per_head]

    using example:
        attention = Self_Attention(30, 2, 50)
        att_outputs = attention(inputs)
    
    注意：
        attention()函数中，最后处理output的时候是针对2层来说的，如果不是两层的self_attention，需要手动该attention函数！！！
'''

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

class Self_Attention(object):
    def __init__(self, size_per_head, embedding_len, nb_head):
        self.embedding_len = embedding_len
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        
    def __call__(self, inputs):
        embedding = self.position_Embedding(inputs, self.embedding_len)
        Q = self.Dense(embedding, self.embedding_len)
        K = self.Dense(embedding, self.embedding_len)
        V = self.Dense(embedding, self.embedding_len)
        output = self.attention(Q, K, V, self.nb_head, self.size_per_head)
        return output

    def position_Embedding(self, inputs, position_size):
        '''
        input: [batch_size, seq_len, word_size]；
        output: [batch_size, seq_len, position_size]。
        '''
        batch_size,seq_len = tf.shape(inputs)[0],tf.shape(inputs)[1]
        position_j = 1. / tf.pow(10000., \
                                2 * tf.range(position_size / 2, dtype=tf.float32 \
                                ) / position_size)
        position_j = tf.expand_dims(position_j, 0)
        position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
        position_i = tf.expand_dims(position_i, 1)
        position_ij = tf.matmul(position_i, position_j)
        position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
        position_embedding = tf.expand_dims(position_ij, 0) \
                            + tf.zeros((batch_size, seq_len, position_size))
        return position_embedding

    def Mask(self, inputs, seq_len, mode='mul'):
        '''
        inputs [batch_size, seq_len, input_size]
        seq_len 每个序列的实际长度
        mode分为mul和add:
            mul是指把多出部分全部置零，一般用于全连接层之前
            add是指把多出部分全部减去一个大的常数，一般用于softmax之前
        '''
        if seq_len == None:
            return inputs
        else:
            mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
            for _ in range(len(inputs.shape)-2):
                mask = tf.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
    
    def Dense(self, inputs, ouput_size, bias=True, seq_len=None):
        '''
        普通的全连接
        inputs: [batch_size,...,input_size]
        output: [batch_size,...,ouput_size]
        '''
        input_size = int(inputs.shape[-1])
        W = tf.Variable(tf.random_uniform([input_size, ouput_size], -0.05, 0.05))
        if bias:
            b = tf.Variable(tf.random_uniform([ouput_size], -0.05, 0.05))
        else:
            b = 0
        outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
        outputs = tf.reshape(outputs, \
                            tf.concat([tf.shape(inputs)[:-1], [ouput_size]], 0)
                            )
        if seq_len != None:
            outputs = self.Mask(outputs, seq_len, 'mul')
        return outputs


    def attention(self, Q, K, V, nb_head, size_per_head, Q_len=None, V_len=None):
        '''
        input:
            Q K V
            nb_head: multi_head 层数
            size_per_head: 输出的维度
        '''
        #对Q、K、V分别作线性映射
        Q = self.Dense(Q, nb_head * size_per_head, False)
        Q = tf.reshape(Q, (-1, tf.shape(Q)[1], nb_head, size_per_head))
        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = self.Dense(K, nb_head * size_per_head, False)
        K = tf.reshape(K, (-1, tf.shape(K)[1], nb_head, size_per_head))
        K = tf.transpose(K, [0, 2, 1, 3])
        V = self.Dense(V, nb_head * size_per_head, False)
        V = tf.reshape(V, (-1, tf.shape(V)[1], nb_head, size_per_head))
        V = tf.transpose(V, [0, 2, 1, 3])
        #计算内积，然后mask，然后softmax
        A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))
        A = tf.transpose(A, [0, 3, 2, 1])
        A = self.Mask(A, V_len, mode='add')
        A = tf.transpose(A, [0, 3, 2, 1])
        A = tf.nn.softmax(A)
        #输出并mask
        O = tf.matmul(A, V)
        O = tf.transpose(O, [0, 2, 1, 3])
        O = tf.reshape(O, (-1, tf.shape(O)[1], nb_head * size_per_head))
        O = self.Mask(O, Q_len, 'mul')

        # nb_head 为2的时候的情况：
        output = (O[:, :, :size_per_head] + O[:, :, size_per_head:2*size_per_head]) / 2
        # nb_head 为3的时候的情况：
        # output = (O[:, :, :size_per_head] + O[:, :, size_per_head:2*size_per_head] 
        #               + O[:, :, size_per_head:3*size_per_head]) / 3

        return output
