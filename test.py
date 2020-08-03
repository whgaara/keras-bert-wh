import time
import re
from tqdm import tqdm


def sss():
    for i in range(1000):
        yield i

for i in tqdm(sss()):
    print(i)


