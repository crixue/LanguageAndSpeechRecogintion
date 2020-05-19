#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform as plat
import os
import numpy as np
import random
#import scipy.io.wavfile as wav
from scipy.fftpack import fft

from utils.wav_preprocess import *
from sklearn.utils import shuffle


def get_thchs30_data(wav_base_path="H:\\PycharmProjects\\dataset\\data_thchs30\\train",
                     trn_base_path="H:\\PycharmProjects\\dataset\\data_thchs30\\data"):
    all_wav_list = shuffle(get_wav_thchs30_list(wav_base_path, trn_base_path))
    return all_wav_list


def data_generator(all_wav_list, batch_size=32):
    '''
    获取thchs30一个batch的数据
    :param all_wav_list: [[wav文件地址list，对应拼音转id的list],...]
    :param batch_size: inputs, outputs
    :return:
    '''
    total_wav_nums = len(all_wav_list)

    for i in range(total_wav_nums//batch_size):
        wav_feature_list = []
        label_data_list = []

        begin = i * batch_size
        end = begin + batch_size

        sub_list = all_wav_list[begin: end]
        for group_data in sub_list:
            fbank = compute_freq_feature(group_data[0])
            fbank = fbank[:fbank.shape[0] // 8 * 8, :]  # 由于声学模型网络结构原因（3个maxpooling层），我们的音频数据的每个维度需要能够被8整除。
            wav_feature_list.append(fbank)
            label_data_list.append(group_data[1])

        padding_wav_data, input_length = wav_data_padding(wav_feature_list)
        padding_label_data, label_length = label_data_padding(label_data_list)

        inputs = {'the_inputs': padding_wav_data,
                  'the_labels': padding_label_data,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc_loss': np.zeros(padding_wav_data.shape[0], )}
        yield inputs, outputs


def wav_data_padding(wav_data_list):
    '''
    每一个batch_size内的数据有一个要求，就是需要构成成一个tensorflow块，这就要求每个样本数据形式是一样的。
    除此之外，ctc需要获得的信息还有输入序列的长度。
    这里输入序列经过卷积网络后，长度缩短了8倍，因此我们训练实际输入的数据为wav_len//8。
    :param wav_data_list:
    :return:
    '''
    wav_len = [len(data) for data in wav_data_list]
    wav_max_len = max(wav_len)
    wav_len = np.array([leng // 8 for leng in wav_len])
    new_wav_data_list = np.zeros((len(wav_len), wav_max_len, 200, 1))
    for i in range(len(wav_data_list)):
        new_wav_data_list[i, :wav_data_list[i].shape[0], :, 0] = wav_data_list[i]
    return new_wav_data_list, wav_len


def label_data_padding(label_data_list):
    '''
    对label进行padding和长度获取，不同的是数据维度不同，且label的长度就是输入给ctc的长度，不需要额外处理
    :param label_data_list:
    :return:
    '''
    label_lens = np.array([len(label) for label in label_data_list])
    max_label_len = max(label_lens)
    new_label_data_lst = np.zeros((len(label_data_list), max_label_len), dtype=np.int32)
    for i in range(len(label_data_list)):
        new_label_data_lst[i][:len(label_data_list[i])] = label_data_list[i]
    return new_label_data_lst, label_lens



if __name__ == '__main__':
    wav_list = get_thchs30_data()
    batch = data_generator(wav_list, 4)
    one_batch = next(batch)
    print()
