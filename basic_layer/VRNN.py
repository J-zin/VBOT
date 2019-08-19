'''
from basic_layer.VRNN import VRNN

method:
    VRNN(z_dim, x_dim, layer_deep, train=True)
        input:
            z_dim: int 输出维度
            x_dim: int 输入维度（实际上可以任意指定维度）
            layer_deep: int 变分和输入数据中的MLP层深度
            train: bool 是否可以训练
        output:
            class VRNN
    
    VRNN.forward(vrnn_input):
        input:
            vrnn_input: 输入数据  [batch_size, sequence_len, embedding_len]
        output:
            vrnn_ouputs: 输出数据 [batch_size, sequence_len, z_dim]
            vrnn_z: 输出的隐藏变量 [batch_size, sequence_len, z_dim]
            kl_loss: KL损失 [batch_size, sequence_len]

using example:
    vrnn = VRNN(int(self.label_dim), int(self.edim), 4) 
    vrnn_ouputs, vrnn_z, kl_loss = vrnn.forward(vrnn_input)

'''


# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

def deepnet(inputs, depth: int, dim: int, name: str, activation=None):
    for i in range(depth):
        inputs = tf.layers.dense(
            inputs, dim, activation=activation, name='%s_%d' % (name, i))
    return inputs

# 为了调整输出使得不报错，h的向量维数变成了prior维数的6倍，不确定是否有更好的方法
# 上述导致h_dim参数是无效的


class VRNNCell(tf.keras.layers.GRUCell):
    def __init__(self,
                 x_dim: int, h_dim: int, z_dim: int,
                 layer_deep: int, train=True, **kwargs):
        '''
        z_dim 输出的mu和sigma的维度
        h_dim RNN隐藏层输出的维度
        x_dim 输入的数据经过MLP后的维度
        '''
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.prior_dim = z_dim
        self.train = train
        self.layer_deep = layer_deep

        super(VRNNCell, self).__init__(units=self.h_dim, **kwargs)
        self.output_size = self.prior_dim*6+self.z_dim

    def call(self, inputs, states, constants=None, **kwargs):
        [h] = states

        with tf.variable_scope(self.get_name(), reuse=tf.AUTO_REUSE):
            prior_hidden = deepnet(
                h, self.layer_deep, self.prior_dim, 'prior_hidden', tf.nn.relu)
            prior_mu = deepnet(prior_hidden, 1,  self.prior_dim, 'prior_mu')
            prior_sigma = deepnet(
                prior_hidden, 1, self.prior_dim, 'prior_sigma', tf.nn.softplus)

            x_hidden = deepnet(inputs, self.layer_deep,
                               self.x_dim, 'x_hidden', tf.sigmoid)
            concat_xh = tf.concat([x_hidden, h], 1)
            enc_hidden = deepnet(
                concat_xh, self.layer_deep, self.prior_dim, 'enc_hidden', tf.nn.relu)
            enc_mu = deepnet(enc_hidden, 1, self.prior_dim, 'enc_mu')
            enc_sigma = deepnet(enc_hidden, 1, self.prior_dim,
                                'enc_sigma', tf.nn.softplus)

            # if self.train:
            #     z_t = tf.truncated_normal(
            #         tf.shape(enc_sigma)) * enc_sigma + enc_mu
            # else:
            #     z_t = enc_mu
            z_t = tf.truncated_normal(
                    tf.shape(enc_sigma)) * enc_sigma + enc_mu
            z_hidden = deepnet(z_t, self.layer_deep,
                               self.z_dim, 'z_hidden', tf.sigmoid)
            concat_xz = tf.concat([x_hidden, z_hidden], 1)
            _, state_next = super(VRNNCell, self).call(
                concat_xz, states, constants, **kwargs)

            concat_zh = tf.concat([z_hidden, h], 1)
            dec_hidden = deepnet(
                concat_zh, self.layer_deep, self.prior_dim, 'dec_hidden', tf.nn.relu)
            dec_mu = deepnet(dec_hidden, 1, self.prior_dim, 'dec_mu')
            dec_sigma = deepnet(dec_hidden, 1, self.prior_dim,
                                'dec_sigma', tf.nn.softplus)

            outputs = [dec_mu, dec_sigma, enc_mu,
                       enc_sigma, prior_mu, prior_sigma, z_t]
            outputs = tf.concat(outputs, axis=1)
        return outputs, state_next

    def build(self, inputs_shape):
        real_shape = inputs_shape.as_list()
        real_shape[-1] = self.z_dim+self.x_dim
        real_shape = tf.TensorShape(real_shape)
        super(VRNNCell, self).build(real_shape)

    def get_z(self, h):
        with tf.variable_scope(self.get_name(), reuse=tf.AUTO_REUSE):
            prior_hidden = deepnet(
                h, self.layer_deep, self.prior_dim, 'prior_hidden', tf.nn.relu)
            prior_mu = deepnet(
                prior_hidden, self.layer_deep,  self.prior_dim, 'prior_mu')

            z = prior_mu
            # z_hidden = deepnet(z, self.layer_deep,
            #                    self.z_dim, 'z_hidden', tf.sigmoid)

        return z

    def set_train(self, train):
        self.train = train

    def get_name(self):
        return type(self).__name__


class VRNN():
    def __init__(self,  z_dim, x_dim, layer_deep, train=True, **kwargs):
        '''
        z_dim 输出的mu和sigma的维度
        x_dim 输入的数据经过MLP后的维度
        h_dim RNN隐藏层输出的维度 默认为 int(z_dim / 2)
        layer_deep 变分和输入数据中的MLP层深度
        '''
        self.layer_deep = layer_deep
        self.h_dim = int(z_dim / 2)
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.train = train
        self.cell = VRNNCell(self.x_dim, self.h_dim, self.z_dim,
                             self.layer_deep)

    def __call__(self, inputs, train=True):
        self.cell.set_train(train)

        outputs, last_state_h = tf.nn.dynamic_rnn(
            cell=self.cell, inputs=inputs, dtype=tf.float32)
        dec_mu = outputs[:, :, 0:1*self.z_dim]
        dec_sigma = outputs[:, :, 1*self.z_dim:2*self.z_dim]
        enc_mu = outputs[:, :, 2*self.z_dim:3*self.z_dim]
        enc_sigma = outputs[:, :, 3*self.z_dim:4*self.z_dim]
        prior_mu = outputs[:, :, 4*self.z_dim:5*self.z_dim]
        prior_sigma = outputs[:, :, 5*self.z_dim:6*self.z_dim]
        z_enc = outputs[:, :, 6*self.z_dim:7*self.z_dim]

        self.z_enc = z_enc
        self.last_state_h = last_state_h
        self.dec_mu = dec_mu
        self.dec_sigma = dec_sigma
        self.enc_mu = dec_mu
        self.enc_sigma = dec_sigma
        self.kl_loss = self.kl_gaussgauss(
            enc_mu, enc_sigma, prior_mu, prior_sigma)
        return self.dec_mu, self.dec_sigma, self.enc_mu, self.enc_sigma, self.kl_loss, self.last_state_h, self.z_enc

    def extra_z(self, h):
        return self.cell.get_z(h)

    def kl_gaussgauss(self, mu_1, sigma_1, mu_2, sigma_2):
        with tf.variable_scope(type(self).__name__+"/kl_gaussgauss"):
            log_sigma_1 = tf.log(tf.maximum(1e-9, sigma_1), name='log_sigma_1')
            log_sigma_2 = tf.log(tf.maximum(1e-9, sigma_2), name='log_sigma_2')
            return tf.reduce_sum(log_sigma_2 - log_sigma_1 + 0.5 * (
                (tf.square(sigma_1)+tf.square(mu_1-mu_2))
                / tf.maximum(1e-9, tf.square(sigma_2)) - 1), 1)

    def forward(self, inputs):
        dec_mu, dec_sigma, enc_mu, enc_sigma, kl_loss, _, z = self.__call__(
                inputs, self.train)

        vrnn_ouputs = (tf.random.truncated_normal(
            tf.shape(dec_sigma)) * dec_sigma + dec_mu)
        vrnn_z = (tf.random.truncated_normal(
            (tf.shape(enc_sigma)[0], 1)) * enc_sigma[:, -1] + enc_mu[:, -1])
        kl_loss = tf.reduce_mean(kl_loss, 1)

        return vrnn_ouputs, vrnn_z, kl_loss

if __name__ == "__main__":
    dim = 5

    inputs = tf.placeholder(tf.float32, [None, 4, 1])
    label = tf.placeholder(tf.float32, [None, 4, 1])

    mu, sigma, kl_loss, _, _, _, _ = VRNN(dim, 10, 10, 4)(inputs)
    # mu, sigma, kl_loss = VRNN(dim, 10, 10, 4, False)(inputs)

    eps = tf.random.truncated_normal(tf.shape(sigma))*sigma+mu
    dense = tf.layers.dense(eps, 1, name="dense1")

    loss = tf.reduce_mean(tf.square(dense-label), 1) + \
        tf.reduce_mean(kl_loss, 1)

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # 预测用
    dense_test = tf.layers.dense(mu, 1, name="dense1", reuse=True)

    init = tf.global_variables_initializer()
    print("finish")
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1000):
            print("=======%d=======" % epoch)
            data_inputs = [[[i*0.1]for i in range(4)]for _ in range(10)]
            data_labels = [[[i*0.1+0.1]for i in range(4)]for _ in range(10)]
            output_loss, output_train, _ = sess.run([loss, dense, optimizer], feed_dict={
                inputs: data_inputs, label: data_labels})

            output_predict = sess.run([dense_test], feed_dict={
                inputs: data_inputs, label: data_labels})
            print(output_predict[0][0], output_loss[0][0])
            # 输出应该接近于0.1,0.2,0.3,0.4,loss趋于0
