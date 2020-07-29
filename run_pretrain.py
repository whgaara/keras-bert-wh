# coding:utf-8
# author:hang.wang

import keras

from pretrain_config import *


class SaveCheckpoint(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(ModelSavePath, overwrite=True)


if __name__ == '__main__':
    save_weights = SaveCheckpoint()
    train = None
    train.fit(
        dataset,
        steps_per_epoch=StepsPerEpoch,
        epochs=Epochs,
        callbacks=[save_weights],
    )
