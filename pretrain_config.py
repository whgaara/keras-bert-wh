# ## 文件路径 ## #
ModelSavePath = 'checkpoint/finetune'
VocabPath = 'checkpoint/pretrain/vocab.txt'

# ## 训练参数 ## #
Epochs = 32
MaskRate = 0.15
BatchSize = 4096
StepsPerEpoch = 10000
# GradAccumSteps大于1表示使用梯度累积
GradAccumSteps = 16
# roberta的段落长度上限为512
SentenceLength = 512
