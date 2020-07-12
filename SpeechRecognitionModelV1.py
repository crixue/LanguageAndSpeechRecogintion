#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform as plat
import os
import time
import logging as log
log.basicConfig(filename='SpeechRecognitionModelV1.log',format='%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S %p',level=log.INFO)

import keras as kr
import numpy as np
import random
import pdb

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization # , Flatten
from keras.layers import Lambda, TimeDistributed, Activation,Conv2D, MaxPooling2D,GRU, Bidirectional #, Merge
from keras.layers.merge import add, concatenate
from keras import backend as K
from keras.optimizers import SGD, Adadelta, Adam

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config=config)
K.set_session(sess)

from utils.common import *
from utils.wav_preprocess import compute_freq_feature
from speech_data import test_data_generator

py2id_dict = pinyin2id()
id2py_dict = dict(zip(py2id_dict.values(), py2id_dict.keys()))

MODEL_LOGS_DIR_NAME = "logs_speechRec_1"

class SpeechRecognitionModelV1():
    '''
    定义CNN/LSTM/CTC模型，使用函数式模型
    输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
    隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2
    隐藏层：全连接层
    输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数，
    CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出

    '''

    def __init__(self):
        '''
        初始化
        默认输出的拼音的表示大小是1423，即1423个拼音+1个空白块
        '''
        self.MS_OUTPUT_SIZE = 1422 + 1 + 1  # 神经网络最终输出的每一个字符向量维度的大小
        # self.BATCH_SIZE = BATCH_SIZE # 一次训练的batch
        self.label_max_string_length = 64
        self.AUDIO_LENGTH = 16000
        self.AUDIO_FEATURE_LENGTH = 200

        self.model, self.ctc_model = self._model_init()

    def _model_init(self):
        input_data = Input(name="the_inputs", shape=(None, self.AUDIO_FEATURE_LENGTH, 1))

        x = Conv2D(32, (3, 3), use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(
            input_data)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=2, strides=None, padding='valid')(x)

        x = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

        x = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=2, strides=None, padding='valid')(x)

        x = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=2, strides=None, padding='valid')(x)

        # 200 / 8 * 128 = 3200
        # 因为DFT抽样点数必须为2的整数次幂，经过3层maxpooling层，要求音频数据的每个维度需要能够被8整除
        x = Reshape(target_shape=(-1, 3200))(x)

        # x = Dropout(0.2)(x)
        x = Dense(128, activation='relu', use_bias=True, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

        gru_units = 128
        # 创建一个双向GRU，看看是否能增加精度？
        #         gru_1a = GRU(gru_units, return_sequences=True, kernel_initializer='he_normal', name='gru_1a')(x)
        #         gru_1b = GRU(gru_units, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru_1b')(x)
        x = Bidirectional(GRU(gru_units, return_sequences=True, kernel_initializer='he_normal', name='gru_1'))(x)
        x = BatchNormalization()(x)

        x = Bidirectional(GRU(gru_units, return_sequences=True, kernel_initializer='he_normal', name='gru_2'))(x)
        x = BatchNormalization()(x)

        x = Dense(128, activation='relu', use_bias=True, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

        x = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(x)

        y_pred = Activation('softmax', name='y_pred_activation')(x)
        model_data = Model(inputs=input_data, outputs=y_pred)

        labels = Input(name='the_labels', shape=[None], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(self._ctc_batch_cost_func, output_shape=(1,), name='ctc_loss')(
            [y_pred, labels, input_length, label_length])

        ctc_model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
        ctc_model.summary()

        optimizer = Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=10e-8)
        ctc_model.compile(optimizer=optimizer, loss={'ctc_loss': lambda y_true, y_pred: y_pred}, metrics=['acc'])

        # captures output of softmax so we can decode the output during visualization
        #         test_func = K.function([input_data], [self.y_pred])
        #         pdb.set_trace()

        # log.info('[*提示] 创建模型成功，模型编译成功')
        log.info('[*Info] Create Model Successful, Compiles Model Successful. ')
        return model_data, ctc_model

    def _ctc_batch_cost_func(self, args):
        y_pred, labels, input_length, label_length = args

        #         pdb.set_trace()
        y_pred = y_pred[:, :, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def predict(self, data_input, input_len):
        '''
        预测结果，返回拼音对应的id列表
        :param data_input: 输入的音频数据
        :param input_len: 音频的长度
        :return:
        '''
        batch_size = 1

        base_pred = self.model.predict(data_input, batch_size)
#         log.info("base_pred-pre:", base_pred)
        base_pred = base_pred[:, :, :]
#         log.info("base_pred:", base_pred)

        r = K.ctc_decode(base_pred, input_len, greedy=True, beam_width=100, top_paths=1)
#         log.info("r:", r)
        r1 = K.get_value(r[0][0])
#         log.info("r1:", r1)

        return r1

    def recognize(self, wav_file):
        '''
        wav 文件识别出拼音
        :param wav_file:
        :return:
        '''
        data_input = compute_freq_feature(wav_file)
        input_length = len(data_input) // 8

        data_input = np.array(data_input, dtype=np.float)
        data_input = np.reshape(data_input.shape[0], data_input.shape[1], 1)

        r1 = self.predict(data_input, input_length)
        recognize_pingyin_list = [id2py_dict[id] for id in r1]
        return recognize_pingyin_list

    def test_model(self, wav_label_list, data_batch_size=4):
        '''
        验证模型的水准
        :param wav_data_path:  验证集wav数据集合
        :param labels_data_path: 对应的拼音labels id集合
        :param data_batch_size: 验证集的数量
        :return:
        '''

        for next_index in range(data_batch_size):
            batch = data_generator(wav_label_list, 1)

            word_error_num = 0
            word_total_num = 0
            input, output = next(batch)

            y_predict = self.predict(input, input['input_length'])
            label = input['the_labels']

            real_str = ''.join([id2py_dict[id] for id in label[0].tolist()])
            predict_str = ''.join([id2py_dict[id] for id in y_predict[0].tolist()])

            words_num = label.shape[1]
            word_total_num += words_num
            distance = calculate_sequence_edit_distance(label[0].tolist(), y_predict[0].tolist())
            if distance <= words_num:
                word_error_num += distance  # 使用编辑距离作为错误字数
            else:  # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字,就直接加句子本来的总字数就好了
                word_error_num += words_num

            log.info("[原本语音内容]：{0}".format(real_str))
            log.info('[**预测结果**]：{0}'.format(predict_str))
            log.info("============********============" + "\n")

        acc = (word_total_num - word_error_num) / word_total_num
        log.info('*本轮语音测试准确率：{}'.format(str(acc)))

    def load_last_weights(self):
        sorted_model_list = sorted(glob.glob(os.path.join(MODEL_LOGS_DIR_NAME, '*.model')),
                                   key=lambda x: time.localtime(os.path.getmtime(x)),
                                   reverse=True)
        if len(sorted_model_list) == 0:
            return
        sorted_ctc_model_list = sorted(glob.glob(os.path.join(MODEL_LOGS_DIR_NAME, '*.model.base')),
                                       key=lambda x: time.localtime(os.path.getmtime(x)),
                                       reverse=True)

        self.model.load_weights(sorted_model_list[0])
        self.ctc_model.load_weights(sorted_ctc_model_list[0])



from utils.wav_preprocess import *
from speech_data import *

train_wav_list = load_various_wav_train_data()
validation_wav_list = load_various_wav_dev_data()


if __name__ == '__main__':
    m = SpeechRecognitionModelV1()
    train_wav_list = train_wav_list[:255000]
    epochs = 100
    batch_size = 5

    batch_num = len(train_wav_list) // batch_size
    # val_batch_num = int(batch_num * 0.2)

    train_batch = data_generator(train_wav_list, batch_size)
    # validation_batch = data_generator(validation_wav_list, batch_size)

    m.load_last_weights()

    for i in range(epochs):
        log.info("Begin epoch:{0}".format(str(i + 1)))
        train_batch = data_generator(train_wav_list, batch_size)
        history = m.ctc_model.fit_generator(train_batch,
                                            steps_per_epoch=batch_num,
                                            epochs=1)
        m.test_model(wav_label_list=validation_wav_list, data_batch_size=64)

        m.model.save_weights(os.path.join(MODEL_LOGS_DIR_NAME, str(i) + '_steps_SpeechRecognitionModelV1.model'))
        m.ctc_model.save_weights(
            os.path.join(MODEL_LOGS_DIR_NAME, str(i) + '_steps_SpeechRecognitionModelV1.model.base'))

