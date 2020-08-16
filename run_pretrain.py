# coding:utf-8
# author:hang.wang

import json
import keras

from model.model import Bert
from pretrain_config import *
from tfrecord_generator import RobertaTrainingData

tfrecords = ['data/roberta_%s.tfrecord' % i for i in range(10)]
data_set = RobertaTrainingData.tfrecord_load(tfrecords)


class SaveCheckpoint(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_checkpoint(ModelSavePath, overwrite=True)


def build_transformer_model(config_path='checkpoint/pretrain/bert_config.json', with_mlm='linear'):
    # 载入bert的配置
    bert_config = json.load(open(config_path))
    if 'sequence_length' not in bert_config:
        bert_config['sequence_length'] = SentenceLength
    bert_config['name'] = 'bert'

    # 加载bert模型
    bert = Bert(**bert_config)
    bert.build()

    return bert, train_model, loss


def build_model_for_pretraining(config_path):
    bert, train_model, loss = build_transformer_model(config_path)

    # 优化器
    # optimizer = extend_with_weight_decay(Adam)
    # if which_optimizer == 'lamb':
    #     optimizer = extend_with_layer_adaptation(optimizer)
    # optimizer = extend_with_piecewise_linear_lr(optimizer)
    # optimizer_params = {
    #     'learning_rate': learning_rate,
    #     'lr_schedule': lr_schedule,
    #     'weight_decay_rate': weight_decay_rate,
    #     'exclude_from_weight_decay': exclude_from_weight_decay,
    #     'bias_correction': False,
    # }
    # if grad_accum_steps > 1:
    #     optimizer = extend_with_gradient_accumulation(optimizer)
    #     optimizer_params['grad_accum_steps'] = grad_accum_steps
    # optimizer = optimizer(**optimizer_params)
    optimizer = None

    # 模型定型
    train_model.compile(loss=loss, optimizer=optimizer)

    return train_model


if __name__ == '__main__':
    save_checkpoint = SaveCheckpoint()
    train_model = build_model_for_pretraining(BertConfigPath)
    dataset = None
    train_model.fit(
        dataset,
        steps_per_epoch=StepsPerEpoch,
        epochs=Epochs,
        callbacks=[save_checkpoint],
    )
