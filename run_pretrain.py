# coding:utf-8
# author:hang.wang

import keras
import tensorflow as tf

from pretrain_config import *
from tfrecord_generator import RobertaTrainingData

tfrecords = ['data/roberta_%s.tfrecord' % i for i in range(10)]
data_set = RobertaTrainingData.tfrecord_load(tfrecords)


class SaveCheckpoint(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_checkpoint(ModelSavePath, overwrite=True)


if __name__ == '__main__':
    save_checkpoint = SaveCheckpoint()
    train = None
    dataset = None
    train.fit(
        dataset,
        steps_per_epoch=StepsPerEpoch,
        epochs=Epochs,
        callbacks=[save_checkpoint],
    )
