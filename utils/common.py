import os
import difflib

def get_symbol_list():
    '''
    读取并返回【拼音-汉字列表】的字典文件
    :param dict_fn:
    :return:
    '''
    dict_fn = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dict.txt")
    dict_symbol = {}
    with open(dict_fn, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            if line == '':
                continue
            combine = line.split('\t')
            pinyin = combine[0]
            chinese_characters = [word for word in combine[1]]
            dict_symbol[pinyin] = chinese_characters
    return dict_symbol


def pinyin2id():
    '''
    返回一个单个拼音对应id的字典
    例如：ai1：6
    :param dict_fn:
    :return:
    '''
    dict_symbol = get_symbol_list()
    single_pinyin_lists = dict_symbol.keys()
    return {single_pinyin: idx for idx, single_pinyin in enumerate(single_pinyin_lists)}


def get_language_model(model_fn):
    '''
    读取并返回语言模型文件
    :param model_fn:
    :return:
    '''

    dict_model = {}
    with open(model_fn, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            if line == '' or len(line) <= 1:
                continue

            combine = line.split('\t')
            dict_model[combine[0]] = combine[1]

    return dict_model


def get_pinyin_model(pinyin_fn):
    '''

    :param pinyin_fn:
    :return:
    '''
    dict_model = {}
    with open(pinyin_fn, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            if line == '':
                continue

            combine = line.split('\t')

            if combine[0] not in dict_model and int(combine[1]) > 1:
                dict_model[combine[0]] = combine[1]
    return dict_model


def cacu_sequence_edit_distance(str1, str2):
    '''
    计算不同字符串之间的文本的距离
    :param str1:
    :param str2:
    :return:
    '''
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        # print('{:7} a[{}: {}] --> b[{}: {}] {} --> {}'.format(tag, i1, i2, j1, j2, str1[i1: i2], str2[j1: j2]))
        if tag == 'replace':
            leven_cost += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            leven_cost += (j2 - j1)
        elif tag == 'delete':
            leven_cost += (i2 - i1)
    return leven_cost
