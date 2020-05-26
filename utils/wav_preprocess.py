import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

from utils.common import *
from sklearn.utils import shuffle


def compute_freq_feature(wav_file):
    '''
    给定一个16khz的音频文件，返回经过离散傅里叶化后的频域数据
    :param wav_file:
    :return:
    '''

    # fs 采样频率
    fs, data = wav.read(wav_file)

    if (16000 != fs):
        raise ValueError(
            '[Error] ASRT currently only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(
                fs) + ' Hz. ')

    N = len(data)  # 取样点数
    Tp = data.shape[0] / fs  # 采样总时长

    time_window = 25
    window_length = int(fs / 1000 * time_window)  # 400

    range0_end = int(Tp * 1000 - time_window) // 10
    data_input = np.zeros((range0_end, window_length // 2), dtype=np.float)  # 用于存放最终的频率特征数据
    data_line = np.zeros((1, window_length), dtype=np.float)
    for i in range(0, range0_end):
        p_begin = i * 160  # 两相邻帧之间有一段重叠区域
        p_end = p_begin + window_length
        frame = data[p_begin: p_end]

        # 加窗
        w = np.hamming(len(frame))
        frame = frame * w

        # 进行快速傅里叶变换
        frame_fft = np.fft.rfft(frame) / N

        # 取对数
        frame_log = np.log(np.abs(frame_fft) + 1)
        #         frame_log =  50 * np.log10(np.clip(np.abs(frame_fft), 1e-20, 1e100))

        data_input[i] = frame_log[:window_length // 2]

    return data_input


def get_wav_thchs30_list(pathname, symbol_data_base_dict):
    '''
    返回一个二元数组，
    每一个元素的第0个元素是wav文件的绝对路径，第1个元素是对应该wav拼音文件转换成id的列表
    :param pathname: thchs30的基本路径
    :param symbol_data_base_dict:  含有对应拼音文件的基本路径
    :return:
    '''
    all_list = []
    for file_abs_path in glob.glob(os.path.join(pathname, "*.wav")):
        file_name = os.path.splitext(os.path.split(file_abs_path)[-1])[0]

        symbol_file_abs_path = os.path.join(symbol_data_base_dict, file_name + ".wav.trn")
        pinyin_str = get_wav_thchs30_symbol(symbol_file_abs_path)
        py_id_list = pinyin_str_trans_to_ids(pinyin_str)

        all_list.append([file_abs_path, py_id_list])

    return all_list


def pinyin_str_trans_to_ids(pinyin_str):
    '''
    将拼音字符串转成id list
    :param pinyin_str:
    :return:
    '''
    py_id_list = []
    pinyin2id_dict = pinyin2id()
    for single_pinyin in pinyin_str.split():
        try:

            if single_pinyin in pinyin2id_dict.keys():
                py_id_list.append(pinyin2id_dict[single_pinyin])
            else:
                single_pinyin += "5"
                py_id_list.append(pinyin2id_dict[single_pinyin])
        except KeyError:
            py_id_list.append("_")
    return py_id_list


def get_wav_thchs30_symbol(file_abs_path):
    '''
    读取指定数据集中，所有wav文件对应的语音符号
    返回一个存储符号集的字典类型值
    '''
    pinyin_str = ''
    with open(file_abs_path, 'r', encoding='utf-8') as f:
        txt_lines = f.readlines()
        if len(txt_lines) < 2:
            pinyin_str = ''
        else:
            pinyin_str = txt_lines[1]

    return pinyin_str


def load_type1_dir_wav_list(wav_root_path):
    '''
    类型1：wav_root_path下有多个wav的子目录
    每个wav文件和wav.trn都在同一个目录下
    :param wav_root_path:
    :return:
    '''
    all_list = []
    for root, dirs, files in tqdm(os.walk(wav_root_path)):
        for file in files:
            if not file.endswith(".wav"):
                continue

            file_abs_path = os.path.join(root, file)
            file_name = os.path.splitext(os.path.split(file)[-1])[0]
            symbol_file_abs_path = os.path.join(root, file_name + ".trn")
            if not os.path.exists(symbol_file_abs_path):
                continue
            pinyin_str = get_wav_thchs30_symbol(symbol_file_abs_path)
            print(symbol_file_abs_path + ":", pinyin_str)
            py_id_list = pinyin_str_trans_to_ids(pinyin_str)

            all_list.append([file_abs_path, py_id_list])
    return all_list


def load_various_wav_train_data():
    all_list = []
    type1_dir = [
        "H:\\PycharmProjects\\dataset\\aidatatang_200zh\\corpus\\train"
    ]
    for part_of_path in type1_dir:
        one_part_wav_list = load_type1_dir_wav_list(part_of_path)
        for one_part_wav in one_part_wav_list:
            all_list.append(one_part_wav)

    part_of_list = get_wav_thchs30_list("H:\\PycharmProjects\\dataset\\data_thchs30\\train",
                                    "H:\\PycharmProjects\\dataset\\data_thchs30\\data")
    for one_part_wav in part_of_list:
        all_list.append(one_part_wav)

    return shuffle(all_list)


def load_various_wav_dev_data():
    all_list = []
    type1_dir = [
        "H:\\PycharmProjects\\dataset\\aidatatang_200zh\\corpus\\dev"
    ]
    for part_of_path in type1_dir:
        one_part_wav_list = load_type1_dir_wav_list(part_of_path)
        for one_part_wav in one_part_wav_list:
            all_list.append(one_part_wav)

    part_of_list = get_wav_thchs30_list("H:\\PycharmProjects\\dataset\\data_thchs30\\dev",
                                    "H:\\PycharmProjects\\dataset\\data_thchs30\\data")
    for one_part_wav in part_of_list:
        all_list.append(one_part_wav)

    return shuffle(all_list)


if __name__ == '__main__':
    all_list = load_various_wav_dev_data()
    print()

