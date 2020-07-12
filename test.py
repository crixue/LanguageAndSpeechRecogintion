import glob
import os
import numpy as np


import glob
import time

MODEL_LOGS_DIR_NAME = "logs_speechRec"

import kenlm

model_path = "D:\\PyWorkspace\\LanguageAndSpeechRecogintion\\data\\zh_giga.no_cna_cmn.prune01244.klm"
model = kenlm.Model(model_path)


def parse_text(text):
    return ' '.join(list(text))


def full_score_acc(s):
    full_score = 0.0
    for prob, _, _ in model.full_scores(s):
        full_score += prob
    return full_score
    # return sum(prob for prob, _, _ in model.full_scores(s))


if __name__ == '__main__':
    print(-1e+6)
    list = {"还", "海", "害"}
    word_list_scores = 0
    total = sum([model.score(word) for word in list])
    print("total:{}".format(total))
    for w in list:
        print("word:{}, acc:{}, per:{}".format(w, model.score(w), model.score(w)/total))

    words0 = "海狮仔"
    words1 = "海事再"
    words2 = "还是栽"


    state = kenlm.State()
    state1 = kenlm.State()
    state2 = kenlm.State()
    state3 = kenlm.State()
    # acc = 0.0
    # model.NullContextWrite(state)
    # acc += model.BaseScore(state, "海", state1)
    # acc += model.BaseScore(state1, "狮", state)
    # acc += model.BaseScore(state, "仔", state1)
    # print(acc)
    # print("====")

    acc = 0.0
    model.NullContextWrite(state)
    acc += model.BaseScore(state, "海", state1)
    acc += model.BaseScore(state1, "狮", state2)
    acc += model.BaseScore(state2, "仔", state3)
    print(acc)
    print("====")
    # Find out-of-vocabulary words
    for w in words0:
        if not w in model:
            print('"{0}" is an OOV'.format(w))
        else:
            score = model.score(w)
            print(score)

    print("====")

    print("====")
    state = kenlm.State()
    state1 = kenlm.State()
    state2 = kenlm.State()
    acc = 0.0
    model.NullContextWrite(state)
    acc += model.BaseScore(state, "海", state1)
    acc += model.BaseScore(state1, "事", state)
    acc += model.BaseScore(state, "再", state1)
    print(acc)
    print("====")
    for w in words1:
        if not w in model:
            print('"{0}" is an OOV'.format(w))
        else:
            score = model.score(w)
            print(score)

    print("====")

    print("====")
    state = kenlm.State()
    state1 = kenlm.State()
    state2 = kenlm.State()
    acc = 0.0
    model.NullContextWrite(state)
    acc += model.BaseScore(state, "还", state1)
    acc += model.BaseScore(state1, "是", state)
    acc += model.BaseScore(state, "在", state1)
    print(acc)
    print("====")
    for w in words2:
        if not w in model:
            print('"{0}" is an OOV'.format(w))
        else:
            score = model.score(w)
            print(score)

    print("====")


    print()