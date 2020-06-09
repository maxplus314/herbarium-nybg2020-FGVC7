import os
import copy
import json
import codecs
import functools
import collections

import pandas as pd
from tensorflow import keras

import augment
import prepare_images
import postprocess_sequence


def load_data(config):
  dir_in = '../input/herbarium-2020-fgvc7/nybg2020'
  dir_pre = '../prepared%dx%d' % config.resolution
  
  path_meta = os.path.join(dir_in, '%s/metadata.json')
  with codecs.open(path_meta % 'train', encoding='utf-8', errors='ignore') \
      as fin:
    train_meta = json.load(fin)
  with codecs.open(path_meta % 'test', encoding='utf-8', errors='ignore') \
      as fin:
    test_meta  = json.load(fin)

  train_an = pd.DataFrame(train_meta['annotations'])
  train_img = pd.DataFrame(train_meta['images'])
  df = train_img.merge(train_an, on='id')
  df.category_id = df.category_id.astype(str)
  idx2class = sorted(frozenset(df.category_id))
  assert len(idx2class) == config.classes

  df = df.sample(frac=1, random_state=1)
  class_entries = collections.Counter(df.category_id)
  df['weight'] = [(1 + min((class_entries[i] - 1) / 4, 9)) / class_entries[i]
                  for i in df.category_id]
  train_size = int(len(df) * .9)
  df_train = df[:train_size]
  df_val = df[train_size:]
  train_steps = (len(df_train) + config.bs - 1) // config.bs
  val_steps = (len(df_val) + config.bs_inf - 1) // config.bs_inf

  dir_in_train = os.path.join(dir_in, 'train')
  dir_pre_train = os.path.join(dir_pre, 'train')
  prepare_images.prepare_images(df, dir_in_train, dir_pre_train,
                                config.resolution, config.workers)
  preprocessor = keras.preprocessing.image.ImageDataGenerator(
      rescale=1 / 255, dtype=config.dtype)
  
  train = preprocessor.flow_from_dataframe(
    df_train,
    x_col='file_name',
    y_col='category_id',
    weight_col='weight',
    directory=dir_pre_train,
    classes=idx2class,
    interpolation='lanczos',
    validate_filenames=False,
    batch_size=config.bs,
    target_size=config.resolution
  )
  val = preprocessor.flow_from_dataframe(
    df_val,
    x_col='file_name',
    y_col='category_id',
    weight_col='weight',
    directory=dir_pre_train,
    classes=idx2class,
    interpolation='lanczos',
    validate_filenames=False,
    batch_size=config.bs_inf,
    target_size=config.resolution
  )
  val_noaugment = preprocessor.flow_from_dataframe(
    df_val,
    x_col='file_name',
    y_col='category_id',
    weight_col='weight',
    directory=dir_pre_train,
    classes=idx2class,
    interpolation='lanczos',
    validate_filenames=False,
    batch_size=config.bs_inf,
    target_size=config.resolution
  )

  df_test = pd.DataFrame(test_meta['images'])
  df_test = df_test.sample(frac=1, random_state=2)
  df_test['dummy_weight'] = [1.] * len(df_test)
  df_test['dummy_category_id'] = [idx2class[0]] * len(df_test)
  test_steps = (len(df_test) + config.bs_inf - 1) // config.bs_inf

  dir_in_test = os.path.join(dir_in, 'test')
  dir_pre_test = os.path.join(dir_pre, 'test')
  prepare_images.prepare_images(df_test, dir_in_test, dir_pre_test,
                                config.resolution, config.workers)
  test = preprocessor.flow_from_dataframe(
    df_test,
    x_col='file_name',
    y_col='dummy_category_id',
    weight_col='dummy_weight',
    directory=dir_pre_test,
    classes=idx2class,
    interpolation='lanczos',
    shuffle=False,
    validate_filenames=False,
    batch_size=config.bs_inf,
    target_size=config.resolution
  )
  
  assert train.class_indices == val.class_indices == \
      val_noaugment.class_indices == test.class_indices == \
      dict(zip(idx2class, range(config.classes)))

  aa = augment.AutoAugment()
  train, val = map(functools.partial(postprocess_sequence.PostprocessSeq,
                                     lambda x: augment.gridcutmix(aa(x))),
      (train, val))

  train, val, val_noaugment, test = map(
      functools.partial(postprocess_sequence.make_enqueuer_generator,
                        workers=config.workers),
      (train, val, val_noaugment, test))

  return (idx2class, train_steps, val_steps, test_steps,
      train, val, val_noaugment, test, df_test.id)
