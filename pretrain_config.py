# ## 文件路径 ## #
ModelSavePath = 'checkpoint/finetune'
VocabPath = 'checkpoint/pretrain/vocab.txt'
BertConfigPath = 'checkpoint/pretrain/bert_config.json'

# ## 训练参数 ## #
Epochs = 32
MaskRate = 0.15
BatchSize = 8
StepsPerEpoch = 16

# roberta的段落长度上限为512
SentenceLength = 512

LearningRate = 1e-4
NumWarmupSteps = 0
# GradAccumSteps大于1表示使用梯度累积
GradAccumSteps = 0
NumTrainSteps = 8
LrSchedule = {
    NumWarmupSteps * GradAccumSteps: 1.0,
    NumTrainSteps * GradAccumSteps: 1.0,
}
WeightDecayRate = 0
ExcludeFromWeightDecay = ['Norm', 'bias']
