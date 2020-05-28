import numpy as np

from utils.common import *


pinyin2word = get_symbol_list()

word2id = word2id()
id2word = dict(zip(word2id.values(), word2id.keys()))

pinyin2id = pinyin2id()
id2pinyin = dict(zip(pinyin2id.values(), pinyin2id.keys()))

# 隐藏状态转移矩阵
trans_total_usage, trans_prob = get_hidden_status_trans_probs()
# min_trans_prob = 0.1 * float(1) / float(trans_total_usage)

# 隐藏状态初始矩阵
hidden_status_total_usage, pi = get_hidden_status_init_probs()
# 没有出现在pi矩阵的字给其设置一个初始的概率
min_word_prob = 0.1 * float(1) / float(hidden_status_total_usage)

# 隐藏状态到观察状态的发射矩阵
emit_probs = get_hidden_to_observer_emit_prob(get_symbol_list())


def get_word_in_pi_prob(word_id):
    '''
    返回某一个隐藏状态（汉字）在初始矩阵的概率，
    如果没有则为min_word_prob
    :param word_id:
    :return:
    '''
    word = id2word[word_id]
    if word in pi.keys():
        return pi.get(word)
    return min_word_prob


def get_pinyin_word_emit_prob(pinyin_word):
    '''
    返回一个隐藏状态到观察状态（字+拼音）组合在发射矩阵的概率，
    如果没有则为min_word_prob
    :param pinyin_word:
    :return:
    '''
    if pinyin_word in emit_probs.keys():
        return emit_probs[pinyin_word]
    return min_word_prob


def viterbi(word_list, pinyin_list, n):
    """
    维特比算法求解最大路径问题
    :param word_list:   每个拼音对应的隐藏状态矩阵
    :param n:   可能观察到的状态数， 对应为汉字数量
    :param id2word:    id到汉字的映射
    :return:
    """
    T = len(word_list)  # 观察状态的长度
    delta = np.zeros((T, n))  # 转移值
    psi = np.zeros((T, n), dtype=int)  # 转移下标值

    # 初始化第一个字符的隐藏初始状态概率， 设置为每个词在词典中的单独出现的概率
    words = word_list[0]
    for i, w in enumerate(words):
        delta[0][i] = get_word_in_pi_prob(w)

    # 动态规划计算
    for idx in range(1, T):
        words = word_list[idx]  # 第T时刻所有可能出现的字的集合
        for i in range(len(words)):
            max_value = 0
            pre_words = word_list[idx-1]

            last_index = 0
            for j in range(len(pre_words)):
                tmp_key = id2word[pre_words[j]] + id2word[words[i]]  # 中国/钟国/忠国
                # 获得转移概率，如果不存在则设置为0
                if tmp_key in trans_prob.keys():
                    prob = trans_prob[tmp_key]
                else:
                    prob = 0

                # 前一时刻的字观察状态到隐藏状态的概率 * 转移概率
                tmp_value = delta[idx-1][j] * prob
                if tmp_value > max_value:
                    max_value = tmp_value
                    last_index = j

            # 计算观察状态到隐藏状态的概率
            tmp_pw_key = id2word[words[i]] + pinyin_list[idx]  # 国guo2
            emit_prob = get_pinyin_word_emit_prob(tmp_pw_key) * max_value  # 观察状态到隐藏状态的概率 * 前一时刻所有字和当前字组合的最大概率

            delta[idx][i] = emit_prob
            psi[idx][i] = last_index  # 保存当前字的前一时刻 和当前字组合的最大概率 的下标值

    prob = 0
    path = np.zeros(T, dtype=int)
    path[T-1] = 1

    # 获取最大的转移值
    desc_word_id = []
    for i in range(n):
        if prob < delta[T-1][i]:
            prob = delta[T-1][i]
            path[T-1] = i
            desc_word_id.append(word_list[T-1][i])

    # 最优路径回溯
    for t in range(T-2, -1, -1):
        last_index = psi[t+1][path[t+1]]
        path[t] = last_index
        desc_word_id.append(word_list[t][last_index])

    final_word = ""
    for id in reversed(desc_word_id):
        final_word += id2word[id]

    return final_word


if __name__ == '__main__':
    pinyin_list = ['qin2', 'lao2', 'yong3', 'gan3']
    pinyin_list = ['zhong1', 'guo2', 'ren2', 'min2']
    pinyin_list = ['xin1', 'zhi1', 'du4', 'ming2']
    word_id_list = []
    n = 0
    for i, single_pinyin in enumerate(pinyin_list):
        single_pinyin_words = pinyin2word[single_pinyin]
        if n < len(single_pinyin_words):
            n = len(single_pinyin_words)
        word_id_list.append([word2id[single_word] for single_word in single_pinyin_words])

    words = viterbi(word_id_list, pinyin_list, n)
    print(words)





