from util.data_api import list_all_fm_file
import numpy as np
import pickle
from dotmap import DotMap

def load_data(filepath):
    inputs, labels, names = data_prepare(filepath)

    data = DotMap()
    data.inputs = inputs
    data.labels = labels
    data.names = names
    
    return data
    
def data_prepare(filepath):
    fm_ys = list_all_fm_file(filepath, 'fm_y')
    inputs = []
    labels = []
    names = []
 
    for i in range(len(fm_ys)):     # 437
        fm_y = fm_ys[i]
        wav_x = fm_ys[i].replace('fm_y', 'wav_x')
        name = fm_ys[i].replace('.fm_y', '').split("/", 2)[2]

        with open(wav_x,'rb') as f:
            result_x = pickle.load(f)
        with open(fm_y,'rb') as f:
            result_y = pickle.load(f)
        # if len(result_x) - len(result_y) > 0:
        #     inputs.append(result_x[:-1, :])
        #     labels.append(result_y)
        # elif len(result_y) - len(result_x) > 0:
        #     inputs.append(result_x)
        #     labels.append(result_y[:-1, :])
        # else:
        #     inputs.append(result_x)
        #     labels.append(result_y)
        inputs.append(result_x[:3600, :])
        labels.append(result_y[:3600, :])
        names.append(name)
    return inputs, labels, names  # input: [437, 3600, 29]    lable: [437, 3600, 30]

def load_test_data(filepath):
    inputs, labels, names = test_data_prepare(filepath)

    data = DotMap()
    data.inputs = inputs
    data.labels = labels
    data.names = names
    
    return data

def test_data_prepare(filepath):
    wav_xs = list_all_fm_file(filepath, 'wav_x')
    inputs = []
    labels = []
    names = []
    print(wav_xs)

    for i in range(len(wav_xs)):     
        wav_x = wav_xs[i]
        name = wav_x.split("/", 2)[2]
        print(name)
        with open(wav_x,'rb') as f:
            result_x = pickle.load(f)
        shape = np.array(result_x).shape
        seq_len = shape[0]
        result_y = np.zeros((seq_len, 30))
        inputs.append(result_x)
        labels.append(result_y)
        names.append(name)
    return inputs, labels, names

# [batch_size, sequence_len, 29] => [batch_size, sequence_len, 15, 29]
def input_prepare(inputs, splite_size):
    shape = np.array(inputs).shape
    batch_size, sequence_len, dims = shape[0], shape[1], shape[2]

    padding = np.ones((batch_size, splite_size-1, dims))
    inputs = np.append(inputs, padding, axis=1)

    result = []
    for i in range(batch_size):
        temp_list = []
        for j in range(sequence_len):
            temp_list.append(inputs[i][j: j+splite_size, :])
        result.append(temp_list)

    # print(np.array(result).shape)
    return result


def non_zero_mean(np_arr):
    exist = (np_arr != 0)
    num = np_arr.sum(axis=0)
    den = exist.sum(axis=0)
    return num/den

# [3571, 30, 30] => [3600, 30, 30]
def output_handle(inputs, stype='mean'):
    """
    input: [dim1, dim2, dim3]   # [3571, 30, 30]
    output: [dim1+dim2-1, dim3]
    """
    shape = np.array(inputs).shape
    dim1, dim2, dim3 = shape[0], shape[1], shape[2]
    seq_len = dim1 + dim2 - 1
    matrix = np.zeros([dim1, seq_len, dim3])
    for i in range(dim1):
        matrix[i, i:i+dim2, :] = inputs[i]
        
    output = []
    for i in range(seq_len):
        temp = matrix[:, i, :]
        # print("temp:")
        # print(temp)
        # print(temp)
        if stype == 'mean':
            # print(np.mean(temp, axis=0))
            temp = non_zero_mean(temp)
            output.append(temp)
        if stype == 'max':
            # print(np.max(temp, axis=0))
            output.append(np.max(temp, axis=0))
    return np.array(output)
