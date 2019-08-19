"""
from basice_layer.LSTMS import single_lstm_layer as lstm_layer

lstm_layer(lstm_input=None, ouput_dim=None, batch_size=None)
    params:
        lstm_input: 数据  [batch_size, sequence_len, embedding_len]
        ouput_dim:  int 输出维度，任意指定  与输入数据维度无关，自动匹配
        batch_size: int batch_size
    output:
        outputs: 输出数据 [batch_size, sequence_len, ouput_dim]
        states: 最后一个隐藏向量 h 的输出   (一般用不到)
"""


# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
    
# 目前看来 单向双层lstm效果较好：multi_lstm_layer() 双向双层和单向双层效果一样：bi_multi_lstm_layer()
def single_lstm_layer(lstm_input=None, ouput_dim=None, batch_size=None):   # 单向单层lstm
    gru_cell = tf.contrib.rnn.BasicLSTMCell(ouput_dim, forget_bias=1.0, state_is_tuple=True)
    # gru_cell = tf.contrib.rnn.DropoutWrapper(cell=gru_cell, input_keep_prob=1.0, output_keep_prob=1.0) # 去掉dropout收敛较快 第一个epoch loss 是20.x
    mgru_cell = tf.contrib.rnn.MultiRNNCell([gru_cell] * 1, state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(mgru_cell, lstm_input, dtype=tf.float32)
    return outputs, states

def multi_lstm_layer(lstm_input, ouput_dim=None, batch_size=None):   # 单向多层lstm  
    # 双层效果好像好一点 第一个epoch loss是 0.067424 收敛在0.004516左右（只跑了30epoch 还在下降）
    cell_list = [tf.nn.rnn_cell.LSTMCell(num_units=ouput_dim) for _ in range(2)]  
    mgru_cell = tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True)     
    outputs, states = tf.nn.dynamic_rnn(mgru_cell, lstm_input, dtype=tf.float32)
    return outputs, states

def bi_lstm_layer(lstm_input, ouput_dim=None, batch_size=None):   # 双向单层lstm  
    # 第一个epoch loss是 0.236033
    cell_fw  = tf.contrib.rnn.BasicLSTMCell(ouput_dim, forget_bias=1.0, state_is_tuple=True)
    cell_bw   = tf.contrib.rnn.BasicLSTMCell(ouput_dim, forget_bias=1.0, state_is_tuple=True)
    init_fw = cell_fw.zero_state(batch_size , dtype=tf.float32)
    init_bw = cell_bw.zero_state(batch_size , dtype=tf.float32)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, lstm_input, initial_state_fw = init_fw, initial_state_bw = init_bw)

    outputs = (outputs[0] + outputs[1]) / 2
    
    return outputs, states

def bi_multi_lstm_layer(lstm_input, ouput_dim=None, batch_size=None):   # 双向多层lstm  
    # 第一个epoch loss是 0.060247
    cells_fw = [tf.nn.rnn_cell.LSTMCell(ouput_dim) for _ in range(2)]
    cells_bw = [tf.nn.rnn_cell.LSTMCell(ouput_dim) for _ in range(2)]
    initial_states_fw = [cell_fw.zero_state(batch_size, tf.float32) for cell_fw in cells_fw]
    initial_states_bw = [cell_bw.zero_state(batch_size, tf.float32) for cell_bw in cells_bw]

    # cells_fw_list = [tf.nn.rnn_cell.LSTMCell(ouput_dim) for _ in range(2)]  
    # cells_fw = tf.contrib.rnn.MultiRNNCell(cells_fw_list, state_is_tuple=True)
    # cells_bw_list = [tf.nn.rnn_cell.LSTMCell(ouput_dim) for _ in range(2)]
    # cells_bw = tf.contrib.rnn.MultiRNNCell(cells_bw_list, state_is_tuple=True) 
    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, lstm_input,
                                                        initial_states_fw=initial_states_fw,
                                                        initial_states_bw=initial_states_bw,
                                                        dtype=tf.float32)

    outputs = outputs[:, :, :ouput_dim]
    # outputs = (outputs[:, :, :ouput_dim] + outputs[:, :, ouput_dim:]) / 2
    return outputs, _