import numpy as np
from dotmap import DotMap
import copy
import random


class batcher(object):
    '''
    seq2seqatt batcher.
    '''

    def __init__(
        self,
        data,
        batch_size=10,
    ):
        self.datas = []
        self.batch_size = batch_size
        self.datas_length = None   # 3600 / 30 = 120
        self.file_num = None    # 437
        self.cur_data = 0       # max = 120-1
        self.cur_batch = 0      # max = 437-1

        data_inputs = np.array(data.inputs)        # (437, 3600, 29)
        data_labels = np.array(data.labels)        # (437, 3600, 30)
        
        shape = np.array(data_inputs).shape
        dim0, dim1 = shape[0], shape[1]
        data_inputs = np.reshape(data_inputs, [int(dim0*3), int(dim1/3), 29])
        data_labels = np.reshape(data_labels, [int(dim0*3), int(dim1/3), 30])
        inputs_labels = np.concatenate((data_inputs, data_labels), axis=2)  # (437*10, 3600/10, 59)
        length = len(inputs_labels[0])               # 437*3

        i = 30
        while(1):
            # batch_input = data_inputs[:, 0:i, :]
            # batch_label = data_labels[:, 0:i, :]
            # batch_data = np.concatenate((batch_input, batch_label), axis=2)
            np.random.shuffle(inputs_labels)
            batch_input = inputs_labels[:, 0:i, :29]
            batch_label = inputs_labels[:, 0:i, 29:]
            self.datas.append({'input': batch_input, 'label': batch_label})

            i += 30
            if i > length:
                break
        
        self.datas_length = len(self.datas)
        self.file_num = len(data_inputs) 
        # print(len(self.datas))      # 3600 / 30 = 120
        # print(np.array(self.datas[0]['input']).shape)   # (437, 30, 29)
        # print(np.array(self.datas[0]['label']).shape)   # (437, 30, 30)
        # print(np.array(self.datas[1]['input']).shape)   # (437, 60, 29)
        # print(np.array(self.datas[1]['label']).shape)   # (437, 60, 30)

    def has_next(self):
        if self.cur_data >= self.datas_length:
            self.cur_batch = 0
            self.cur_data = 0
            return False
        else:
            return True

    def next_batch(self):
        if not self.has_next():
            print("use has_next() to identify if has next batch before use next_batch() to get next batch!!!")
            exit()

        inputs = self.datas[self.cur_data]['input']
        labels = self.datas[self.cur_data]['label']
        if self.cur_batch + self.batch_size < self.file_num:
            data_input = inputs[self.cur_batch: self.cur_batch+self.batch_size, :, :]
            data_label = labels[self.cur_batch: self.cur_batch+self.batch_size, :, :]
            self.cur_batch += self.batch_size
            return data_input, data_label
        else:
            # print("cur_batch: ", self.cur_batch)
            # print("labels_len: ", len(labels))
            data_input = inputs[self.cur_batch: , :, :]
            data_label = labels[self.cur_batch: , :, :]
            self.cur_batch = 0
            self.cur_data += 1
            return data_input, data_label


