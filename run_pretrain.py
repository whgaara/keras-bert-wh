# coding:utf-8
# author:hang.wang

import json
import tensorflow.compat.v1 as tf #使用1.0版本的方法
tf.disable_v2_behavior()
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

from model.model import Bert
from model.optimizers import *
from keras.layers import *
from keras.models import Model
from pretrain_config import *
from tfrecord_generator import RobertaTrainingData

tfrecords = ['data/roberta_%s.tfrecord' % i for i in range(10)]
data_set = RobertaTrainingData.tfrecord_load(tfrecords)


class SaveCheckpoint(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_checkpoint(ModelSavePath, overwrite=True)


def build_transformer_model(config_path='checkpoint/pretrain/bert_config.json'):
    token_ids = keras.layers.Input(shape=(None,), dtype='int64', name='token_ids')
    is_masked = keras.layers.Input(shape=(None,), dtype='float64', name='is_masked')

    # 载入bert的配置
    bert_config = json.load(open(config_path))
    if 'sequence_length' not in bert_config:
        bert_config['sequence_length'] = SentenceLength
    bert_config['name'] = 'bert'
    # 加载bert模型
    bert = Bert(**bert_config)
    bert.build()
    bert_outputs = bert.outputs[0]

    def mlm_loss(inputs):
        y_gt, y_prob, y_mask = inputs
        y_gt = K.cast(y_gt, 'float32')
        y_mask = K.cast(y_mask, 'float32')
        loss = K.sparse_categorical_crossentropy(y_gt, y_prob, from_logits=True)
        loss = K.sum(loss * y_mask) / (K.sum(y_mask) + K.epsilon())
        return loss

    # def mlm_acc(inputs):
    #     y_gt, y_prob, y_mask = inputs
    #     y_true = K.cast(y_gt, 'float32')
    #     acc = keras.metrics.sparse_categorical_accuracy(y_true, y_prob)
    #     acc = K.sum(acc * y_mask) / (K.sum(y_mask) + K.epsilon())
    #     return acc

    mlm_loss = Lambda(mlm_loss, name='mlm_loss')([token_ids, bert_outputs, is_masked])
    # mlm_acc = Lambda(mlm_acc, name='mlm_acc')([token_ids, bert_outputs, is_masked])

    # load weights
    # if checkpoint_path is None:

    train_model = Model(bert.inputs + [token_ids, is_masked], [mlm_loss])

    loss = {
        'mlm_loss': lambda y_true, y_pred: y_pred,
        # 'mlm_acc': lambda y_gt, y_prob: K.stop_gradient(y_prob),
    }

    return train_model, loss


def build_model_for_pretraining(config_path):
    train_model, loss = build_transformer_model(config_path)

    # 优化器
    # optimizer = extend_with_weight_decay(Adam)
    # optimizer = extend_with_piecewise_linear_lr(optimizer)
    # optimizer_params = {
    #     'LearningRate': LearningRate,
    #     'LrSchedule': LrSchedule,
    #     'WeightDecayRate': WeightDecayRate,
    #     'ExcludeFromWeightDecay': ExcludeFromWeightDecay,
    #     'bias_correction': False,
    # }
    # if GradAccumSteps > 1:
    #     optimizer = extend_with_gradient_accumulation(optimizer)
    #     optimizer_params['grad_accum_steps'] = GradAccumSteps
    # optimizer = optimizer(**optimizer_params)

    # 模型定型
    # train_model.compile(loss=loss, optimizer=optimizer)
    train_model.compile(optimizer='adam', loss=loss)

    return train_model


if __name__ == '__main__':
    save_checkpoint = SaveCheckpoint()
    train_model = build_model_for_pretraining(BertConfigPath)
    train_model.fit(
        data_set,
        steps_per_epoch=StepsPerEpoch,
        epochs=Epochs,
        callbacks=[save_checkpoint]
    )
