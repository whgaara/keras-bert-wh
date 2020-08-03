import re
import json
import glob
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from model.tokenizers import Tokenizer
from pretrain_config import *


def get_texts():
    filenames = glob.glob('data/*.txt')
    np.random.shuffle(filenames)
    count, texts = 0, []
    for filename in filenames:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                texts.append(line)
                count += 1
                if count == 10:  # 10篇文章合在一起再处理
                    yield texts
                    count, texts = 0, []
    if texts:
        yield texts


class RobertaTrainingData(object):
    def __init__(self, record_name):
        self.tokenizer = Tokenizer(VocabPath, do_lower_case=True)
        self.vocab_size = self.tokenizer._vocab_size
        self.token_pad_id = self.tokenizer._token_pad_id
        self.token_cls_id = self.tokenizer._token_start_id
        self.token_sep_id = self.tokenizer._token_end_id
        self.token_mask_id = self.tokenizer._token_mask_id
        self.writer = tf.python_io.TFRecordWriter(record_name)

    def token_process(self, token_id):
        """
        以80%的几率替换为[MASK]，以10%的几率保持不变，
        以10%的几率替换为一个随机token。
        """
        rand = np.random.random()
        if rand <= 0.8:
            return self.token_mask_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0, self.vocab_size)

    def texts_to_ids(self, texts):
        texts_ids = []
        for text in texts:
            words_tokes = self.tokenizer.tokenize(text=text)
            # text_ids首位分别是cls和sep，这里暂时去除
            words_ids = self.tokenizer.tokens_to_ids(words_tokes)[1:-1]
            texts_ids.append(words_ids)
        return texts_ids

    def ids_to_mask(self, texts_ids):
        """
        这里只对每个字做了mask，其实还可以考虑先对句子进行分词，如果是一个词的，可以对词中所有字同时进行mask
        """
        instances = []

        total_ids = []
        total_masks = []

        total_ids.append(self.token_cls_id)
        total_masks.append(self.token_pad_id)

        for text_ids in texts_ids:
            # 每个实例句子最长保留512，-2是给cls和sep留位置
            text_ids = text_ids[:SentenceLength-2]
            # 为每个字生成对应概率
            mask_rates = np.random.random(len(texts_ids))
            for i, word_id in enumerate(text_ids):
                total_ids.append(word_id)
                if mask_rates[i] < MaskRate:
                    total_masks.append(self.token_process(word_id))
                else:
                    total_masks.append(0)
            # 判断当前total是否已经达到最大长度
            # if len(total_ids) >




if __name__ == '__main__':
    robert = RobertaTrainingData('robert.tfrecord')
    for texts in tqdm(get_texts()):
        texts_ids = robert.texts_to_ids(texts)
        robert.ids_to_mask(texts_ids)
        print(1)
