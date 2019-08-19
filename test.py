import pickle
import numpy as np

def read_fmy():    
    f = open('2019_0627_151718.fm_y','rb')
    fm_y = pickle.load(f)
    print(fm_y)
    print(np.array(fm_y).shape)

    f = open('2019_0627_151718-true.fm_y','rb')
    fm_y = pickle.load(f)
    print(fm_y)
    print(np.array(fm_y).shape)

    # f = open('2019_0627_151718.fm_y','wb')
    # fm_y = np.matrix(fm_y[0])
    # pickle.dump(fm_y, f)

def test_shuffle():
    test = [
        [
            [1, 2, 3],
            [1, 1]
        ],
        [
            [1, 2, 3, 4],
            [1,1,1]
        ],
        [
            [2],
            [3,3]
        ]

    ]
    print(test)                 # [[[1, 2, 3], [1, 1]], [[1, 2, 3, 4], [1, 1, 1]], [[2], [3, 3]]]
    np.random.shuffle(test)
    print(test)                 # [[[1, 2, 3], [1, 1]], [[2], [3, 3]], [[1, 2, 3, 4], [1, 1, 1]]]

def test_concate():
    test1 = [
        [
            [1, 2, 3],
            [1, 1, 1]
        ],
        [
            [2, 3, 4],
            [1, 1, 1]
        ]
    ]
    test2 = [
        [
            [1, 2, 3],
            [1, 1, 1]
        ],
        [
            [2, 3, 4],
            [1, 1, 1]
        ]
    ]
    print(np.array(test1).shape)
    print(np.array(test2).shape)
    resutl = np.concatenate((test1, test2), axis=0)
    print(np.array(resutl).shape)

def test_dsplit():
    test1 = np.array([
        [
            [1, 2, 3, 4],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2,2,2,2]
        ],
        [
            [2, 3, 4, 5],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2,3,4,5]
        ]
    ])
    print(np.array(test1).shape) # [2,4,4]
    result = test1.reshape(4, 2, 4)
    print(result)


if __name__ == "__main__":
    test_dsplit()

