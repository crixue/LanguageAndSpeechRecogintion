import os
import difflib
import tarfile as tar

from tqdm import tqdm
from pypinyin import pinyin, lazy_pinyin, Style

dict_txt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dict.txt")
lang_model1_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "language_model1.txt")
lang_model2_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "language_model2.txt")


def get_hidden_status_trans_probs():
    '''
    返回隐藏状态转移矩阵
    例如：{'中国': 0.00423,...}
    :return:
    '''
    trans_prob = {}
    with open(lang_model2_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        total_usage = lines[0]
        for line in lines[1:]:
            if line == '':
                continue
            combine = line.split('\t')
            phrase = combine[0]
            usage = combine[1]
            trans_prob[phrase] = float(usage) / float(total_usage)
    return total_usage, trans_prob


def get_hidden_status_init_probs():
    '''
    隐藏状态初始矩阵
    例如：{'的':0.054, '一':0.018, ...}
    :return:
    '''
    init_prob = {}
    with open(lang_model1_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        total_usage = lines[0]
        for line in lines[1:]:
            if line == '':
                continue
            combine = line.split('\t')
            word = combine[0]
            usage = combine[1]
            init_prob[word] = float(usage) / float(total_usage)
    return total_usage, init_prob


def get_hidden_to_observer_emit_prob(pinyin_word_dict):
    '''
    隐藏状态到观察状态的发射矩阵
    {"周zhou":0.5,"粥zhou":0.1,}
    :param get_symbol_list():
    :return:
    '''
    emit_prob = {}
    _ , init_prob = get_hidden_status_init_probs()
    for word, usage in init_prob.items():
        mutli_pys = pinyin(word, style=Style.TONE3, heteronym=True)
        if len(mutli_pys) > 0:
            for single_one in mutli_pys[0]:
                if single_one in pinyin_word_dict.keys():
                    emit_prob[word + single_one] = usage
    return emit_prob



def get_symbol_list():
    '''
    读取并返回【拼音-汉字列表】的字典文件
    :param dict_fn:
    :return:
    '''
    dict_symbol = {}
    with open(dict_txt_path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            if line == '':
                continue
            combine = line.split('\t')
            pinyin = combine[0]
            chinese_characters = [word for word in combine[1] if word != '\n']
            dict_symbol[pinyin] = chinese_characters
    return dict_symbol


def word2id():
    '''
    返回一个汉字对应id的列表
    例如：啊：6
    :return:
    '''
    dict_symbol = get_symbol_list()
    words_list = dict_symbol.values()
    word_dict = {}
    count = 0
    for words in words_list:
        for word in words:
            word_dict[word] = count
            count += 1
    return word_dict


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
    读取并返回拼音到汉字列表的映射
    例如{'a1': '阿啊呵腌吖锕'}
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


def calculate_sequence_edit_distance(str1, str2):
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


def un_tar(file_dir):
    '''
    解压文件夹下面的所有tar.gz文件到本地目录下
    :param file_dir:
    :return:
    '''
    for root, dirs, files in os.walk(file_dir):
        for file in tqdm(files):
            if not file.endswith(".tar.gz"):
                continue

            with tar.open(os.path.join(root, file)) as tf:
                names = tf.getnames()
                # 循环解压缩，将压缩文件中的所有文件解压缩
                for name in names:
                    # print(name)
                    tf.extract(name, root)


def chinese_chars_transform_pingyin(chars):
    '''
    中文转拼音
    :param chars: 中国
    :return: e.g. ['zhong1', 'guo2']
    '''
    return lazy_pinyin(chars, style=Style.TONE3)


def write_pinyin_to_trn(file_dir):
    for root, dirs, files in tqdm(os.walk(file_dir,topdown=True)):
        for file in files:
            if not file.endswith(".trn"):
                continue

            with open(os.path.join(root, file), mode='r+', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) < 1:
                    continue
                chars = lines[0].split(' ')  # 字幕文件的第一行是文字字幕
                if len(chars) < 1:
                    continue
                pinyin_list = [pys for pys in chinese_chars_transform_pingyin(chars)]
                pinyin_str = ' '.join(pinyin_list)
                print(pinyin_str)
                f.write("\n" + pinyin_str)


if __name__ == '__main__':
    # dict_model = get_hidden_status_init_probs()
    write_pinyin_to_trn("H:\\PycharmProjects\\dataset\\aidatatang_200zh\\corpus")
    print()