import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import activations
from tensorflow.keras import initializers


def conv(filters, dropout_rate, config, kernel=3, strides=1, pool=False, norelu=False):
  def f(x):
    padding = 'same' if strides == 1 or config.pad else 'valid'
    initializer = initializers.VarianceScaling(2, mode='fan_avg',
                                               distribution='normal')
    c = layers.Conv2D(filters, kernel, padding=padding, strides=strides,
                      kernel_initializer=initializer, dtype=config.policy)(x)
    if dropout_rate:
      c = layers.Dropout(dropout_rate, noise_shape=[None, 1, 1, None],
                         dtype=config.policy)(c)
    c = layers.BatchNormalization(dtype=tf.float32)(c)
    if not norelu:
      c = layers.ReLU(dtype=config.policy)(c)
    if pool:
      c = layers.MaxPooling2D(dtype=config.policy)(c)
    return c
  return f


def cutfirst(config, ratio=4):
  def g(x):
    return x[:, :, :, :x.shape[-1] // ratio]

  def f(x):
    return layers.Lambda(g, dtype=config.policy)(x)
  return f


def dense(size, config, dropout_rate=0, activation='relu'):
  if activation == 'relu':
    activation = layers.ReLU(dtype=config.policy)
  def f(x):
    initializer = initializers.VarianceScaling(2, mode='fan_avg',
                                               distribution='normal')
    
    d = layers.Dense(size, kernel_initializer=initializer,
                     dtype=config.policy)(x)
    if dropout_rate:
      d = layers.Dropout(dropout_rate, dtype=config.policy)(d)
    d = layers.BatchNormalization(dtype=tf.float32)(d)
    if activation is not None:
      d = activation(d)
    return d
  return f


def squeeze_excitation(config):
  """
    Motivated by 'Squeeze-and-Excitation Networks'
    [arxiv.org/abs/1709.01507]
  """
  def f(x):
    p = layers.GlobalAveragePooling2D(dtype=config.policy)(x)
    filters = int(p.shape[1])
    d0 = dense(filters // 4, config)(p)
    d1 = dense(filters, config, activation=layers.Lambda(
        activations.sigmoid, dtype=config.policy))(d0)
    d1 = layers.Reshape((1, 1, -1), dtype=config.policy)(d1)
    return layers.Multiply(dtype=config.policy)([x, d1])
  return f


def stride(config):
  def g(x):
    stride_shift = 1 - config.pad
    strides = 2
    return x[:, stride_shift::strides, stride_shift::strides]

  def f(x):
    return layers.Lambda(g, dtype=config.policy)(x)
  return f


def dense_block(filters, dropout_rate, config, shrink=False):
  """
    Inspired by 'Densely Connected Convolutional Networks'
    [arxiv.org/abs/1608.06993]
  """
  def f(x):
    strides = 1 + config.strided * shrink
    first_kernel = 7 if x.shape[-1] == 3 else 1
    c0 = conv(filters, dropout_rate, config, kernel=first_kernel)(x)
    c0 = layers.Concatenate(dtype=config.policy)([c0, x])

    c1 = conv(filters * strides, dropout_rate, config, strides=strides)(
        cutfirst(config)(c0))
    if strides != 1:
      c0 = stride(config)(c0)
    c1 = layers.Concatenate(dtype=config.policy)([c1, c0])

    c2 = conv(filters * strides, dropout_rate, config, kernel=1)(c1)
    c2 = squeeze_excitation(config)(c2)
    c2 = layers.Concatenate(dtype=config.policy)([c2, c1])
    if not config.strided and shrink:
      c2 = layers.MaxPooling2D(dtype=config.policy)(c2)
    return c2
  return f


def stage(filters, dropout_rate, config, shrink=True):
  def f(x):
    c0 = dense_block(filters, dropout_rate, config)(x)
    c1 = dense_block(filters, dropout_rate, config, shrink=shrink)(c0)
    return c1
  return f

  
def condense(config):
  def f(x):
    d = layers.GlobalAveragePooling2D(dtype=config.policy)(x)
    d = layers.BatchNormalization(dtype=tf.float32)(d)
    return d
  return f


def create_model(config):
  x = layers.Input((*config.resolution, config.channels), dtype=config.dtype)
  dropout_rate = .1 / np.log2(config.resolution_scale)
  base_neurons = 2**10 // config.resolution_scale
  c1 = stage(base_neurons * 2**0, dropout_rate, config)(x)
  c2 = stage(base_neurons * 2**1, dropout_rate * 2, config)(c1)
  c3 = stage(base_neurons * 2**2, dropout_rate * 3, config)(c2)
  c4 = stage(base_neurons * 2**3, dropout_rate * 4, config)(c3)

  clast = c4
  assert clast.shape[1:3] == config.resolution_top
  d = condense(config)(clast)
  d = dense(config.classes, config)(d)
  y = layers.Activation('softmax', dtype=tf.float32)(d)
  model = models.Model(x, y)
  return model
