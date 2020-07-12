import kenlm
import os
import logging as log
log.basicConfig(filename='SpellCheck.log',
                format='%(asctime)s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S %p',
                level=log.INFO)
from pycorrector import Corrector


class SpellCheckModel():

    def __init__(self):
        # model = kenlm.Model(os.path.join(os.path.dirname(__file__), 'data', 'zh_giga.no_cna_cmn.prune01244.klm'))
        self.model = Corrector(language_model_path=os.path.join(os.path.dirname(__file__), 'data', 'zh_giga.no_cna_cmn.prune01244.klm'))
        pass

    def correct_cn_words(self, words):
        corrected_sent, detail = self.model.correct(words)
        # log.info('ori words:{0}; corrected words:{1}; detail:{2}'.format(words, corrected_sent, detail))
        print('ori words:{0}; corrected words:{1}; detail:{2}'.format(words, corrected_sent, detail))
        return corrected_sent


if __name__ == '__main__':
    sp_model = SpellCheckModel()
    corrected_sent = sp_model.correct_cn_words('金天转了好多钱')
    print(corrected_sent)