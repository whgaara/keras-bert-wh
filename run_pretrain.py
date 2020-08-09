# coding:utf-8
# author:hang.wang

import keras

from pretrain_config import *
from tfrecord_generator import RobertaTrainingData

tfrecords = ['data/roberta_%s.tfrecord' % i for i in range(10)]
data_set = RobertaTrainingData.tfrecord_load(tfrecords)


class SaveCheckpoint(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_checkpoint(ModelSavePath, overwrite=True)


def build_model_for_pretraining():
    train_model = None
    return train_model


if __name__ == '__main__':
    save_checkpoint = SaveCheckpoint()
    train_model = build_model_for_pretraining()
    dataset = None
    train_model.fit(
        dataset,
        steps_per_epoch=StepsPerEpoch,
        epochs=Epochs,
        callbacks=[save_checkpoint],
    )
