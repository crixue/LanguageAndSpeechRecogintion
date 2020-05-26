import glob
import os
import numpy as np


import glob
import time

MODEL_LOGS_DIR_NAME = "logs_speechRec"
if __name__ == '__main__':
    # un_tar("D:\\pyworkspace\\data\\train-test\\A2_0")
    # write_pinyin_to_trn("D:\\pyworkspace\\data\\train-test\\A2_0")
    # print(chinese_chars_transform_pingyin("中国"))

    sorted_model_list = sorted(glob.glob(os.path.join(MODEL_LOGS_DIR_NAME, '*.model')),
                               key=lambda x: time.localtime(os.path.getmtime(x)),
                               reverse=True)
    print()