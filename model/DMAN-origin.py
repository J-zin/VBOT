import pandas as pd
import numpy as np
from basic_layer.NN_adam import NN
from util.batcher import batcher
from basic_layer.VRNN import VRNN
from basic_layer.LSTMs import multi_lstm_layer as lstm_layer
from basic_layer.Self_Attention import Self_Attention
from data.datahandle import input_prepare
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
            [None, None, self.edim],  # [batch_sie, sequence_len, edim]
            name="inputs"
        )

        self.labels = tf.placeholder(
            tf.float32,
            [None, None, self.label_dim],   # [bathc_size, sequence_len, label_dim]
            name="labels"
        )

        self.tensor_batch_size = tf.placeholder(tf.int32, (), name='batch_size')  # 给lstm专用

        return self.inputs, self.labels

    def vrnn_layer(self, vrnn_input):
        vrnn = VRNN(int(self.label_dim), int(self.edim), 4) 
        vrnn_ouputs, vrnn_z, kl_loss = vrnn.forward(vrnn_input)
        
        return vrnn_ouputs, vrnn_z, kl_loss
    
    def self_atte_layer(self, inputs):
        attention = Self_Attention(self.label_dim, 50, 2)
        att_outputs = attention(inputs)
        return att_outputs

    def build_model(self):
        inputs, labels = self.set_placeholder()

        # ====== build your own model ======
        self.vrnn_ouputs = self.self_atte_layer(inputs)

        # self.vrnn_ouputs, _ = lstm_layer(inputs, self.label_dim, self.tensor_batch_size)
        # self.vrnn_ouputs, self.vrnn_z, self.kl_loss = self.vrnn_layer(inputs)


        self.l2loss = tf.norm(self.vrnn_ouputs- labels)/tf.norm(labels)
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
                
                
                batch_input, batch_label = bt.next_batch()
                batch_input = input_prepare(batch_input, 15)
                feed_dict = {
                    self.inputs: batch_input,
                    self.labels: batch_label,
                    self.tensor_batch_size: len(batch_input)
                }
                crt_loss, optimize = sess.run([self.loss, self.optimize], feed_dict=feed_dict)
                loss.append(crt_loss)

            mean_loss = np.mean(loss)
            print("\nEpoch{}\ttain-loss: {:.6f}".format(t_round, mean_loss))
            if min_loss > mean_loss:
                min_loss = mean_loss
                self.save_model(sess, self.model_save_path, self.model_name, saver)
            print("testing....")
            self.test(sess, test_data, saver)

        # 未分割数据 [1, 3600, 29] 搭建模型时 可以用这个 速度较快
        # data_inputs = train_data.inputs
        # data_labels = train_data.labels
        # min_loss = 9999
        # for t_round in range(self.epoch):
        #     loss = []
        #     for i in range(len(data_inputs)):
        #         batch_input = np.expand_dims(data_inputs[i], 0)
        #         batch_label = np.expand_dims(data_labels[i], 0)
        #         feed_dict = {
        #             self.inputs: batch_input,
        #             self.labels: batch_label,
        #             self.tensor_batch_size: len(batch_input)
        #         }
        #         crt_loss, optimize = sess.run([self.loss, self.optimize], feed_dict=feed_dict)
        #         loss.append(crt_loss)

        #     mean_loss = np.mean(loss)
        #     print("\nEpoch{}\ttain-loss: {:.6f}".format(t_round, mean_loss))
        #     if min_loss > mean_loss:
        #         min_loss = mean_loss
        #         self.save_model(sess, self.model_save_path, self.model_name, saver)
        #     print("testing....")
        #     self.test(sess, test_data, saver)

    def test(self, sess, test_data, saver):
        data_inputs = test_data.inputs
        data_labels = test_data.labels
        data_names = test_data.names

        loss = []
        for i in range(len(data_inputs)):
            batch_input = np.expand_dims(data_inputs[i], 0)
            batch_label = np.expand_dims(data_labels[i], 0)
            name = data_names[i]
            
            feed_dict = {
                self.inputs: batch_input,
                self.labels: batch_label,
                self.tensor_batch_size: len(batch_input)
            }
            crt_loss, fm_y = sess.run([self.loss, self.vrnn_ouputs], feed_dict=feed_dict)
            if crt_loss < self.test_loss[i]:
                self.test_loss[i] = crt_loss
                super(DMAN, self).save_fmy(sess, self.fmy_output_path, fm_y, name, self.gpu_num, self.model_name)
            
            loss.append(crt_loss)
        print("test-loss: {:.6f}".format(np.mean(loss)))
