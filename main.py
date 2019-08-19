import sys, os
import numpy as np
import pandas as pd
from optparse import OptionParser
from model.DMAN import *
from util.config import read_conf
from util.tensorflow_api import get_session
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

from data.datahandle import load_data
from data.datahandle import load_test_data

def option_parse():
    parser = OptionParser()
    parser.add_option(
        "-g",
        "--gpunum",
        action='store',
        type='string',
        dest="gpunum",
        default="0"
    )
    parser.add_option(
        "-a",
        "--nottrain",
        action='store_true',
        dest="not_train",
        default=False
    )
    parser.add_option(      # 载入模型的路径
        "-p",
        "--modelpath",
        action='store',
        type='string',
        dest="model_path",
        default='output/model/201908061717-DMAN.dman'
    )
    parser.add_option(      # 模型保存的名字
        "-n",
        "--modelname",
        action='store',
        type='string',
        dest="model_name",
        default='DMAN'
    )
    (option, args) = parser.parse_args()
    return option

def main(options):
    gpu_num = options.gpunum
    is_train = not options.not_train            # default: true
    model_path = options.model_path             # 载入模型的路径
    model_name = options.model_name             # 模型保存的名字


    config = read_conf('dman', 'config/model.conf')
    config['gpu_num'] = gpu_num
    config['is_train'] = is_train
    config['model_path'] = model_path       # 载入模型的路径
    config['model_name'] = model_name       # 模型保存的名字

    train_data = load_data(config['dataset_path'])
    test_data = load_test_data(config['test_dataset_path'])

    with tf.Graph().as_default():
        dman = DMAN(config)
        dman.build_model()

        saver = tf.train.Saver(max_to_keep=30)

        sess = get_session(gpu_num=config['gpu_num'], gpu_fraction=0.95)
        init = tf.global_variables_initializer()
        sess.run(init)
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())

        if is_train:
            dman.train(sess, train_data, test_data, saver)
        else:
            print("--------------------test only, no train-------------------------")
            saver.restore(sess, model_path)
            dman.test(sess, test_data, saver)

        print("success!!!")

if __name__ == "__main__":
    options = option_parse()
    main(options)