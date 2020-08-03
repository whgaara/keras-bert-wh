# ## 文件路径 ## #
ModelSavePath = 'checkpoint/finetune'
VocabPath = 'checkpoint/pretrain/vocab.txt'

# ## 训练参数 ## #
Epochs = 32
StepsPerEpoch = 10000
# roberta的段落长度上限为512
MaskRate = 0.15
SentenceLength = 512
