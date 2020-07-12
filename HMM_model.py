import numpy as np
import kenlm
from utils.common import *


pinyin2word = get_symbol_list()

word2id = word2id()
id2word = dict(zip(word2id.values(), word2id.keys()))

pinyin2id = pinyin2id()
id2pinyin = dict(zip(pinyin2id.values(), pinyin2id.keys()))


class HMM_Model():
    def __init__(self):
        # 隐藏状态转移矩阵
        self.trans_total_usage, self.trans_prob = get_hidden_status_trans_probs()

        # 隐藏状态初始矩阵
        self.hidden_status_total_usage, self.pi = get_hidden_status_init_probs()
        # 没有出现在pi矩阵的字给其设置一个初始的概率
        self.min_word_prob = 0.1 * float(1) / float(self.hidden_status_total_usage)

        # 隐藏状态到观察状态的发射矩阵
        self.emit_probs = get_hidden_to_observer_emit_prob(get_symbol_list())

        self.model_path = os.path.join(os.path.dirname(__file__), 'data', 'zh_giga.no_cna_cmn.prune01244.klm')
        self.model = kenlm.Model(self.model_path)
        self.min_prob = -1e+3
        pass

    def get_word_in_pi_prob(self, word_id):
        '''
        返回某一个隐藏状态（汉字）在初始矩阵的概率，
        如果没有则为min_word_prob
        :param word_id:
        :return:
        '''
        word = id2word[word_id]
        if word in self.pi.keys():
            return self.pi.get(word)
        return self.min_word_prob


    def get_pinyin_word_emit_prob(self, pinyin_word):
        '''
        返回一个隐藏状态到观察状态（字+拼音）组合在发射矩阵的概率，
        如果没有则为min_word_prob
        :param pinyin_word:
        :return:
        '''
        if pinyin_word in self.emit_probs.keys():
            return self.emit_probs[pinyin_word]
        return self.min_word_prob


    def viterbi(self, word_list, pinyin_list, n):
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
            delta[0][i] = self.get_word_in_pi_prob(w)

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
                    if tmp_key in self.trans_prob.keys():
                        prob = self.trans_prob[tmp_key]
                    else:
                        prob = 0

                    # 前一时刻的字观察状态到隐藏状态的概率 * 转移概率
                    tmp_value = delta[idx-1][j] * prob
                    if tmp_value > max_value:
                        max_value = tmp_value
                        last_index = j

                # 计算观察状态到隐藏状态的概率
                tmp_pw_key = id2word[words[i]] + pinyin_list[idx]  # 国guo2
                emit_prob = self.get_pinyin_word_emit_prob(tmp_pw_key) * max_value  # 观察状态到隐藏状态的概率 * 前一时刻所有字和当前字组合的最大概率

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


    def get_word_in_pi_prob_use_kenlm(self, word_id):
        '''
        返回某一个隐藏状态（汉字）在初始矩阵的概率，
        如果没有则为min_word_prob
        :param word_id:
        :return:
        e.g.
        word:还, acc:-6.3978071212768555
        '''
        word = id2word[word_id]
        if not word in self.model:
            return self.min_prob
        return self.model.score(word)


    def get_hidden_to_observer_emit_prob_use_kenlm(self, pinyin, word):
        '''
        隐藏状态到观察状态（字+拼音）组合的发射矩阵
        :param pinyin:guo2
        :param word: 国
        :return:
        e.g.
        word:害, acc:-5.514978885650635, per:0.31204276487005345
        word:海, acc:-5.761005878448486, per:0.32596320675334983
        word:还, acc:-6.3978071212768555, per:0.36199402837659667
        '''
        word_list = pinyin2word[pinyin]
        single_word_score = self.model.score(word)
        total = sum([self.model.score(w) for w in word_list])
        return single_word_score/total


    def get_trans_prob_use_kenlm(self, *args):
        '''
        获取转移概率, 使用log函数做处理，分数越接近0，转移概率越高
        :param phrase: 词组类似：中国/钟国/忠国
        :return: -9.256282567977905
        '''
        word_list = args
        state = kenlm.State()
        state1 = kenlm.State()
        self.model.NullContextWrite(state)

        acc = 0.0
        for index, word in enumerate(word_list):
            if index % 2 == 0:
                acc += self.model.BaseScore(state, word, state1)
            else:
                acc += self.model.BaseScore(state1, word, state)
        return acc


    def viterbi_use_kenlm_model(self, word_list, pinyin_list, n):
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
            delta[0][i] = abs(self.get_word_in_pi_prob_use_kenlm(w))

        # 动态规划计算
        for idx in range(1, T):
            words = word_list[idx]  # 第T时刻所有可能出现的字的集合
            for i in range(len(words)):
                # max_value = -1e+4
                max_value = 1e+4
                pre_words = word_list[idx-1]

                last_index = 0
                for j in range(len(pre_words)):
                    tmp_key = id2word[pre_words[j]] + id2word[words[i]]  # 中国/钟国/忠国
                    # 获得转移概率
                    prob = abs(self.get_trans_prob_use_kenlm(id2word[pre_words[j]], id2word[words[i]]))

                    # 前一时刻的字观察状态到隐藏状态的概率 * 转移概率
                    tmp_value = delta[idx-1][j] * prob
                    # if tmp_value > max_value:
                    #     max_value = tmp_value
                    #     last_index = j

                    if tmp_value < max_value:
                        max_value = tmp_value
                        last_index = j

                # 计算观察状态到隐藏状态的概率
                emit_prob = abs(self.get_hidden_to_observer_emit_prob_use_kenlm(pinyin_list[idx], id2word[words[i]])) * max_value  # 观察状态到隐藏状态的概率 * 前一时刻所有字和当前字组合的最大概率

                delta[idx][i] = emit_prob
                psi[idx][i] = last_index  # 保存当前字的前一时刻 和当前字组合的最大概率 的下标值

        prob = 1e+4
        path = np.zeros(T, dtype=int)
        path[T-1] = 1

        # 获取最大的转移值
        desc_word_id = []
        fit_index = 0
        for i in range(n):
            if delta[T-1][i] != 0 and prob > delta[T-1][i]:
                prob = delta[T-1][i]
                path[T-1] = i
                fit_index = i
        desc_word_id.append(word_list[T-1][fit_index])

        # 最优路径回溯
        for t in range(T-2, -1, -1):
            last_index = psi[t+1][path[t+1]]
            path[t] = last_index
            desc_word_id.append(word_list[t][last_index])

        final_word = ""
        for id in reversed(desc_word_id):
            final_word += id2word[id]

        return final_word


    def viterbi_use_kenlm_model_v1(self, word_list, pinyin_list, n):
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
            delta[0][i] = abs(self.get_word_in_pi_prob_use_kenlm(w))

        # 动态规划计算
        for idx in range(1, T):
            words = word_list[idx]  # 第T时刻所有可能出现的字的集合
            for i in range(len(words)):
                max_value = 1e+4
                pre_words = word_list[idx-1]

                last_index = 0
                for j in range(len(pre_words)):
                    if idx > 1:  # 3n-gram
                        pre_pre_word = word_list[idx-2]
                        for k in range(len(pre_pre_word)):
                            tmp_key = id2word[pre_pre_word[k]] + id2word[pre_words[j]] + id2word[words[i]]  # 中国地/种过地
                            # 获得转移概率
                            prob = abs(self.get_trans_prob_use_kenlm(id2word[pre_pre_word[k]], id2word[pre_words[j]], id2word[words[i]]))
                            # 前一时刻的字观察状态到隐藏状态的概率 * 转移概率
                            tmp_value = delta[idx - 1][j] * prob

                            if tmp_value < max_value:
                                max_value = tmp_value
                                last_index = j
                    else:  # 2n-gram
                        tmp_key = id2word[pre_words[j]] + id2word[words[i]]  # 果地/过地/国地
                        # 获得转移概率
                        prob = abs(self.get_trans_prob_use_kenlm(id2word[pre_words[j]], id2word[words[i]]))
                        # 前一时刻的字观察状态到隐藏状态的概率 * 转移概率
                        tmp_value = delta[idx - 1][j] * prob

                        if tmp_value < max_value:
                            max_value = tmp_value
                            last_index = j

                # 计算观察状态到隐藏状态的概率
                emit_prob = abs(self.get_hidden_to_observer_emit_prob_use_kenlm(pinyin_list[idx], id2word[words[i]])) * max_value  # 观察状态到隐藏状态的概率 * 前一时刻所有字和当前字组合的最大概率

                delta[idx][i] = emit_prob
                psi[idx][i] = last_index  # 保存当前字的前一时刻 和当前字组合的最大概率 的下标值

        prob1 = 1e+4
        path = np.zeros(T, dtype=int)
        path[T-1] = 1

        # 获取最大的转移值
        desc_word_id = []
        fit_index = 0
        for i in range(n):
            if delta[T-1][i] != 0 and prob1 > delta[T-1][i]:
                prob1 = delta[T-1][i]
                path[T-1] = i
                fit_index = i
        desc_word_id.append(word_list[T-1][fit_index])

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
    tests = [
        ['zhong1', 'guo2', 'ren2', 'min2'],
        ['ming2', 'tian1', 'gao1', 'kao3'],
        ['hou4', 'tian1', 'jiu4', 'yao4', 'chu1', 'cheng2', 'ji4', 'le5'],
        ['xi1', 'wang4', 'ni3', 'hou4', 'tian1', 'gao1', 'kao3', 'shun4', 'li4'],
        ['wo3', 'ming2', 'tian1', 'yao4', 'chi1', 'ping2', 'guo3'],
        ['ni3', 'de5', 'bao4', 'jia4', 'tai4', 'gao1', 'le5'],
        ['qin2', 'lao2', 'yong3', 'gan3'],
        ['xin1', 'zhi1', 'du4', 'ming2'],
        ['jin1', 'tian1', 'zhuan4', 'le5', "hao3", 'duo1', 'qian2'],
        ['ni3', 'shi4', 'fou3', 'you3', "hao3", 'duo1', 'wen4', 'hao4']
    ]

    for pinyin_list in tests:
        word_id_list = []
        n = 0
        for i, single_pinyin in enumerate(pinyin_list):
            single_pinyin_words = pinyin2word[single_pinyin]
            if n < len(single_pinyin_words):
                n = len(single_pinyin_words)
            word_id_list.append([word2id[single_word] for single_word in single_pinyin_words])

        model = HMM_Model()
        words = model.viterbi_use_kenlm_model_v1(word_id_list, pinyin_list, n)
        print(words)





