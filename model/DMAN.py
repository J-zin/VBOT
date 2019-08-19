import pandas as pd
import numpy as np
from basic_layer.NN_adam import NN
from util.batcher import batcher
from basic_layer.VRNN import VRNN
from basic_layer.LSTMs import bi_lstm_layer as lstm_layer
from basic_layer.Self_Attention import Self_Attention
from data.datahandle import input_prepare
from data.datahandle import output_handle
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

class DMAN(NN):
    def __init__(self, config):
        super(DMAN, self).__init__(config)
        self.test_loss = [9999] * 20
        if config != None:
            self.edim = config['edim']
            self.label_dim = config['label_dim']
            self.epoch = config['nepoch']
            self.fmy_output_path = config['fmy_output_path']
            self.model_save_path = config['model_save_path']
            self.batch_size = config['batch_size']
            self.gpu_num = config['gpu_num']
            self.model_name = config['model_name']
        else:
            self.edim = None
            self.label_dim = None
            self.epoch = None
            self.fmy_output_path = None
            self.model_save_path = None
            self.batch_size = None
            self.gpu_num = None
            self.model_name = None

    def set_placeholder(self):
        self.inputs = tf.placeholder(
            tf.float32,
            [None, None, 15, self.edim],  # [batch_sie, sequence_len, 15, 29]
            name="inputs"
        )

        self.labels = tf.placeholder(
            tf.float32,
            [None, None, self.label_dim],   # [bathc_size, sequence_len, label_dim]
            name="labels"
        )

        shape = tf.shape(self.inputs)
        batch_size = shape[0]
        sequence_len = shape[1]
        
        reshape_inputs = tf.reshape(self.inputs, [batch_size, sequence_len, -1, 3, self.edim]) # [batch_sie, sequence_len, 5, 3, edim]

        return reshape_inputs, self.labels

    def lstm_layer1(self, inputs):
        """
        input: [batch_sie, sequence_len, 5, 3, 29]
        output: [batch_sie, sequence_len, 5, 3, 64]
        """
        with tf.variable_scope('layer1'):
            shape = tf.shape(inputs)    # inputs: [batch_sie, sequence_len, 5, 3, 29]
            num1, num2, num3 = shape[0], shape[1], shape[2]

            inputs = tf.reshape(inputs, [num1*num2*num3, -1, 29])
            lstm_output, _ = lstm_layer(inputs, 64, num1*num2*num3)
            lstm_output = tf.reshape(lstm_output, [num1, num2, num3, -1, 64])

        return lstm_output          # outputs: [batch_sie, sequence_len, 5, 3, 64]
    
    def attn_layer1(self, inputs):
        """
        input: [batch_sie, sequence_len, 5, 3, 64]
        output: [batch_sie, sequence_len, 5, 64]
        """
        with tf.variable_scope('layer1'):
            shape = inputs.get_shape().as_list()
            batch_size, sequence_len, split_num, dim = shape[0], shape[1], shape[2], shape[4]
            output = tf.layers.dense(inputs, units=dim, activation='tanh')          # output: [batch_sie, sequence_len, 5, 3, 64]
            output = tf.nn.softmax(tf.layers.dense(output, units=1, activation=None))  # output: [batch_sie, sequence_len, 5, 3, 1]
            output = tf.matmul(output, inputs, transpose_a=True)                    # output: [batch_sie, sequence_len, 5, 1, 64]
            output = tf.squeeze(output, squeeze_dims=3)                             # output: [batch_sie, sequence_len, 5, 64]
            
        return output

    def lstm_layer2(self, inputs):
        """
        input: [batch_sie, sequence_len, 5, 64]
        ouput: [batch_sie, sequence_len, 5, 128]
        """
        # print(inputs.get_shape())
        with tf.variable_scope('layer2'):
            shape = tf.shape(inputs)
            dim1, dim2, dim3 = shape[0], shape[1], shape[2]

            inputs = tf.reshape(inputs, [dim1*dim2, dim3, 64])    # [batch_sie * sequence_len, 5, 64]
            lstm_output, _ = lstm_layer(inputs, 128, dim1*dim2)
            lstm_output = tf.reshape(lstm_output, [dim1, dim2, dim3, 128])
            
        return lstm_output

    def attn_layer2(self, inputs):
        """
        input: [batch_sie, sequence_len, 5, 128]
        ouput: [batch_sie, sequence_len, 128]
        """
        with tf.variable_scope('layer2'):
            shape = inputs.get_shape().as_list()
            dim1, dim2, dim3, dim4 = shape[0], shape[1], shape[2], shape[3]
            output = tf.layers.dense(inputs, units=dim4, activation='tanh')         # output: [batch_sie, sequence_len, 5, 128]
            output = tf.nn.softmax(tf.layers.dense(output, units=1, activation=None))   # output: [batch_sie, sequence_len, 5, 1]
            output = tf.matmul(output, inputs, transpose_a=True)                   # output: [batch_sie, sequence_len, 1, 128]
            output = tf.squeeze(output, squeeze_dims=2)                             # output: [batch_sie, sequence_len, 128]

        return output

    def lstm_layer3(self, inputs):
        """
        input: [batch_sie, sequence_len, 256]
        output: [batch_sie, sequence_len, 256]
        """
        with tf.variable_scope('layer3'):
            shape = tf.shape(inputs)
            batch_size = shape[0]

            lstm_output, _ = lstm_layer(inputs, 256, batch_size)   

        return lstm_output
    
    def mlp_layer(self, inputs):
        """
        input: [batch_sie, sequence_len, 256]
        output: [batch_sie, sequence_len, 30]
        """
        with tf.variable_scope('top_layer'):
            output = tf.layers.dense(inputs, units=128, activation='tanh')  # [batch_sie, sequence_len, 128]
            output = tf.layers.dense(output, units=30, activation=None)     # [batch_sie, sequence_len, 30]
        
        return output

    def build_model(self):
        inputs, labels = self.set_placeholder()

        # ====== build your own model ======

        layer_output = self.lstm_layer1(inputs)               # output: [batch_sie, sequence_len, 5, 3, 64]
        layer_output = self.attn_layer1(layer_output)   # output: [batch_sie, sequence_len, 5, 64]
        layer_output = self.lstm_layer2(layer_output)   # output: [batch_sie, sequence_len, 5, 128]
        layer_output = self.attn_layer2(layer_output)   # output: [batch_sie, sequence_len, 128]
        layer_output = self.lstm_layer3(layer_output)   # output: [batch_sie, sequence_len, 256]
        layer_output = self.mlp_layer(layer_output)           # output: [batch_sie, sequence_len, 30]
        self.model_output = layer_output

        self.l2loss = tf.reduce_mean(tf.square(self.model_output - labels) * 100)
        # self.l2loss = (tf.norm(self.model_output - labels) / (tf.norm(labels)))
        # self.l2loss = 1000 * tf.reduce_mean(tf.square(self.model_output - labels))
        self.loss = self.l2loss

        # ====== build your own model ======


        self.params = tf.trainable_variables()
        self.optimize, self.islfGrad = super(DMAN, self).optimize_normal(
            self.loss, self.params)


    def train(self, sess, train_data, test_data, saver):

        # 分割数据
        bt = batcher(train_data, self.batch_size)
        print("-------------begin train------------------")
        min_loss = 9999
        for t_round in range(self.epoch):
            loss = []
            while(1):
                if not bt.has_next():
                    break
                
                batch_input, batch_label = bt.next_batch()      # [batch_size, dequnce_len, 29]
                batch_input = input_prepare(batch_input, 15)    # [batch_size, sequence_len, 15, 29]

                shape = np.array(batch_input).shape
                batch_size = shape[0]
                seq_len = shape[1]
                # print(np.array(batch_input).shape)
                # print(np.array(batch_label).shape)
                # if seq_len % 300 == 0 and batch_size % 10 != 0:
                #     print("-----------------batch_input shape----------------------")
                #     print(np.array(batch_input).shape)
                #     print("-----------------batch_input shape----------------------")
                feed_dict = {
                    self.inputs: batch_input,
                    self.labels: batch_label
                }
                crt_loss, optimize = sess.run([self.loss, self.optimize], feed_dict=feed_dict)
                # print(gradient)
                # print(crt_loss)
                print("loss: {}".format(crt_loss))
                print("facial feature: ")
                print(batch_label[0][0])
                # print("--------------------------------")
                loss.append(crt_loss)


            mean_loss = np.mean(loss)
            print("\nEpoch{}\ttain-loss: {:.6f}".format(t_round, mean_loss))
            if min_loss > mean_loss:
                min_loss = mean_loss
                self.save_model(sess, self.model_save_path, self.model_name, saver)
                print("save model...")
                self.test(sess, test_data, saver)

    def test(self, sess, test_data, saver):
        data_inputs = test_data.inputs
        data_labels = test_data.labels
        data_names = test_data.names

        for i in range(len(data_inputs)):
            sequence_len = len(data_inputs[i])
            batch_input = np.expand_dims(data_inputs[i], 0)     # [1, sequence_len(3600), 29]
            batch_label = np.expand_dims(data_labels[i], 0)
            name = data_names[i]

            inputs = [input_prepare(batch_input[:, j:j+30, :], 15)[0] for j in range(sequence_len-29)]  # [3571, 30, 15, 29]
            labels = [batch_label[:, j:j+30, :][0] for j in range(sequence_len-29)] # [3571, 30, 30]
            # print(np.array(inputs).shape)
            # print(np.array(labels).shape)
            feed_dict = {
                self.inputs: inputs,                   # [3571, 30, 15, 29]
                self.labels: labels                    # [3571, 30, 30]
            }
            fm_y = sess.run([self.model_output], feed_dict=feed_dict)   # [1, 3571, 30, 30]
            # print(fm_y)
            fm_y = output_handle(fm_y[0])
            print(name)
            super(DMAN, self).save_fmy(sess, self.fmy_output_path, fm_y, name, self.gpu_num, self.model_name)
