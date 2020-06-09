import multiprocessing

import tensorflow as tf


class Config(object):
  def __init__(self, replicas):
    super(Config, self).__init__()
    self.bs = 32 * replicas
    self.bs_inf = 256 * replicas
    self.bs_inf_mem = 2**13 // self.bs_inf
    
    self.strided = True
    self.pad = True if self.strided else None
    shrink_delta = self.strided and 2 * self.pad - 1
    self.resolution_scale = 2**4
    height_top = 8 + shrink_delta
    width_top = 5 + shrink_delta
    height = (height_top - shrink_delta) * self.resolution_scale + shrink_delta
    width = (width_top - shrink_delta) * self.resolution_scale + shrink_delta
    self.resolution_top = (height_top, width_top)
    self.resolution = (height, width)
    
    self.channels = 3
    self.classes = 32093
    
    self.workers = multiprocessing.cpu_count()
    use_mixed_precision = False
    if use_mixed_precision:
      self.dtype = tf.float16
      self.policy = tf.keras.mixed_precision.experimental.Policy(
          'mixed_float16', loss_scale='dynamic')
    else:
      self.dtype = self.policy = tf.float32
