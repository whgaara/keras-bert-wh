import re
import json
import glob
import numpy as np


def get_texts():
    filenames = glob.glob('pretrain/*.txt')
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


class TrainingData(object):
    pass


if __name__ == '__main__':
    st = get_texts()
    for i in st:
        print(i)
