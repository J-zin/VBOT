import numpy as np

def non_zero_mean(np_arr):
    exist = (np_arr != 0)
    num = np_arr.sum(axis=0)
    den = exist.sum(axis=0)
    return num/den

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

def output_handle_all(inputs, stype='mean'):
    """
    input: [dim1, dim2, dim3]
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
        delete_list = []
        for j in range(dim1):
            delete = True
            for k in range(dim3):
                if temp[j][k] != 0:
                    delete = False
            if delete:
                delete_list.append(j)
        temp = np.delete(temp, delete_list, axis=0)
        # print(temp)
        if stype == 'mean':
            # print(np.mean(temp, axis=0))
            output.append(np.mean(temp, axis=0))
        if stype == 'max':
            # print(np.max(temp, axis=0))
            output.append(np.max(temp, axis=0))
        if stype == 'min':
            # print(np.min(temp, axis=0))
            output.append(np.min(temp, axis=0))
        if stype == 'median':
            # print(np.median(temp, axis=0))
            output.append(np.median(temp, axis=0))
    
    return np.array(output)

def test_none_zero():
    test = [
        [1, 1, 1],
        [2, 2, 2],
        [0, 0, 0],
        [0, 0, 0]
    ]
    print(non_zero_mean(np.array(test)))

if __name__ == "__main__":
    # data = [
    #     [
    #         [1, 1, 1],
    #         [1, 1, 1]
    #     ], [
    #         [2, 2, 2],
    #         [3, 3, 3]
    #     ]
    # ]
    # # print(np.array(data).shape)
    # output = output_handle(data, 'mean')
    # print(np.array(output).shape)
    # print(output)
    
    test_none_zero()