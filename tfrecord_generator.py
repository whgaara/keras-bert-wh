import os
import glob
import math
import pkuseg
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
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                texts.append(line)
                count += 1
                # 10个句子组成一个段落
                if count == 10:
                    yield texts
                    count, texts = 0, []
    if texts:
        yield texts


class RobertaTrainingData(object):
    def __init__(self):
        self.tokenizer = Tokenizer(VocabPath, do_lower_case=True)
        self.seg = pkuseg.pkuseg()
        self.vocab_size = self.tokenizer._vocab_size
        self.token_pad_id = self.tokenizer._token_pad_id
        self.token_cls_id = self.tokenizer._token_start_id
        self.token_sep_id = self.tokenizer._token_end_id
        self.token_mask_id = self.tokenizer._token_mask_id

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
            # 处理每个句子
            # 注意roberta里并不是针对每个字进行mask，而是对字或者词进行mask
            words = self.seg.cut(text)
            for word in words:
                # text_ids首位分别是cls和sep，这里暂时去除
                word_tokes = self.tokenizer.tokenize(text=word)[1:-1]
                words_ids = self.tokenizer.tokens_to_ids(word_tokes)
                texts_ids.append(words_ids)
        return texts_ids

    def ids_to_mask(self, texts_ids):
        """
        这里只对每个字做了mask，其实还可以考虑先对句子进行分词，如果是一个词的，可以对词中所有字同时进行mask
        """
        instances = []
        total_ids = []
        total_masks = []
        # 为每个字或者词生成一个概率，用于判断是否mask
        mask_rates = np.random.random(len(texts_ids))

        for i, word_id in enumerate(texts_ids):
            # 为每个字生成对应概率
            total_ids.extend(word_id)
            if mask_rates[i] < MaskRate:
                # 因为word_id可能是一个字，也可能是一个词
                for sub_id in word_id:
                    total_masks.append(self.token_process(sub_id))
            else:
                total_masks.extend([0]*len(word_id))

        # 每个实例的最大长度为512，因此对一个段落进行裁剪
        # 510 = 512 - 2，给cls和sep留的位置
        for i in range(math.ceil(len(total_ids)/510)):
            tmp_ids = [self.token_cls_id]
            tmp_masks = [self.token_pad_id]
            tmp_ids.extend(total_ids[i*510: min((i+1)*510, len(total_ids))])
            tmp_masks.extend(total_masks[i*510: min((i+1)*510, len(total_masks))])
            # 不足512的使用padding补全
            diff = SentenceLength - len(tmp_ids)
            if diff == 1:
                tmp_ids.append(self.token_sep_id)
                tmp_masks.append(self.token_pad_id)
            else:
                # 添加结束符
                tmp_ids.append(self.token_sep_id)
                tmp_masks.append(self.token_pad_id)
                # 将剩余部分padding补全
                tmp_ids.extend([self.token_pad_id] * (diff - 1))
                tmp_masks.extend([self.token_pad_id] * (diff - 1))
            instances.append([tmp_ids, tmp_masks])
        return instances

    def tfrecord_serialize(self, instances):
        instance_keys = ['token_ids', 'mask_ids']
        serialize_instances = []
        for instance in instances:
            features = {k: tf.train.Feature(int64_list=tf.train.Int64List(value=v))
                        for k, v in zip(instance_keys, instance)}
            tf_features = tf.train.Features(feature=features)
            tf_example = tf.train.Example(features=tf_features)
            serialize_instance = tf_example.SerializeToString()
            serialize_instances.append(serialize_instance)
        return serialize_instances


if __name__ == '__main__':
    robert = RobertaTrainingData()
    # 考虑到使用动态mask，因此，同样的句子会重复10次，每次mask的内容是不一样的。
    for i in range(10):
        writer = tf.python_io.TFRecordWriter(os.path.join('data', 'roberta_%s.tfrecord' % i))
        for texts in tqdm(get_texts()):
            texts_ids = robert.texts_to_ids(texts)
            instances = robert.ids_to_mask(texts_ids)
            serialize_instances = robert.tfrecord_serialize(instances)
            for serialize_instance in serialize_instances:
                writer.write(serialize_instance)
