import sys
import time
import contextlib

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
import tensorflow as tf
import tensorflow_addons as tfa

import adamw
import load_data
import train_config
import create_model
import progress_logger

class DummyScopeStrategy(object):
  def scope(self):
    return self

  def __enter__(self):
    return self

  def __exit__(self, *args):
    pass


class Validate(callbacks.Callback):
  def __init__(self, val, steps):
    super(Validate, self).__init__()
    self.val = val
    self.steps = steps
    self.val_ev = None
  
  def flush(self):
    if self.val_ev:
      print('val', *zip(self.model.metrics_names, self.val_ev))
  
  def on_epoch_begin(self, epoch, logs=None):
    self.flush()
  
  def on_train_end(self, epoch, logs=None):
    self.flush()
  
  def on_epoch_end(self, epoch, logs=None):
    self.val_ev = self.model.evaluate(self.val, steps=self.steps, verbose=0,
                                      workers=0, max_queue_size=0)


def main():
  tf.config.optimizer.set_jit(True)
  
  strategy = tf.distribute.MirroredStrategy()
  replicas = strategy.num_replicas_in_sync
  print('replicas:', replicas)
  if replicas == 1:
    strategy = DummyScopeStrategy()
  config = train_config.Config(replicas)
  idx2class, train_steps, val_steps, test_steps, \
      train, val, val_noaugment, test, test_ids = load_data.load_data(config)

  with strategy.scope():
    reduce_lr = callbacks.ReduceLROnPlateau(
      monitor='val_categorical_accuracy',
      factor=0.5,
      patience=6,
      cooldown=0,
      min_lr=3e-6,
      min_delta=0.002,
      verbose=1
    )
    model = create_model.create_model(config)
    optimizer = adamw.AdamW(lr=1e-3, weight_decay=3e-5,
                            steps_per_epoch=train_steps)
    model.compile(
      optimizer=optimizer,
      loss='categorical_crossentropy',
      weighted_metrics=['categorical_accuracy'],
      metrics=[tfa.metrics.F1Score(config.classes, 'macro')]
    )

  prob_test = np.zeros((len(test_ids), config.classes), dtype=np.float16)

  callback_list = [reduce_lr, Validate(val, val_steps)]
  """
    Horizontal voting motivated by Horizontal and Vertical Ensemble
    with Deep Representation for Classification [arxiv.org/abs/1306.2759]
  """
  for test_sample in range(10):
    initial_epoch = 0
    epochs = 100 + test_sample
    validation_data = val_noaugment
    validation_steps = val_steps
    steps_per_epoch = train_steps
    testing = test_sample > 0
    if testing:
      initial_epoch = epochs - 1
      validation_data = None
      validation_steps = None
      steps_per_epoch = train_steps // 5
      callback_list = []
    try:
      model.fit(
        train,
        validation_data=validation_data,
        epochs=epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1,
        callbacks=callback_list,
        workers=0,
        max_queue_size=1
      )
    except KeyboardInterrupt:
      pass
    try:
      logger = progress_logger.ProgressLogger('predict', test_steps)
      for i, batch in zip(range(test_steps), test):
        prob_test[i * config.bs_inf:(i + 1) * config.bs_inf] += \
            model.predict_on_batch(batch[0])
        logger.update_log()
    except KeyboardInterrupt:
      break
    print()

  y_test = np.argmax(prob_test, axis=1)
  pred = [idx2class[i] for i in y_test]
  df_pred = pd.DataFrame({'Id' : test_ids, 'Predicted' : pred})
  df_pred.to_csv('submission.csv', index=False)
  
  with open('predicted_probs', 'wb') as fout:
    for megabatch in range(0, test_steps, config.bs_inf_mem):
      fout.write(prob_test[megabatch * config.bs_inf:
          (megabatch + config.bs_inf_mem) * config.bs_inf].tobytes())


if __name__ == '__main__':
  main()
