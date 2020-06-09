import tensorflow as tf
from tensorflow.keras.optimizers import Adam


class AdamW(Adam):
  """Motivated by 'Decoupled Weight Decay Regularization' [arxiv.org/abs/1711.05101]"""
  def __init__(self, weight_decay=0.01, steps_per_epoch=1, epochs=1, **kwargs):
    super(AdamW, self).__init__(**kwargs)
    self.wdscale = weight_decay * (1 / steps_per_epoch / epochs)**.5
  
  def _resource_apply_dense(self, grad, var, apply_state=None):
    op = super(AdamW, self)._resource_apply_dense(grad, var, apply_state)
    wd = tf.compat.v1.assign_sub(var, var * self.wdscale)
    return tf.group([op, wd])
  
  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    raise NotImplementedError('Only dense update supported')
    return super(AdamW, self)._resource_apply_sparse(
        grad, var, indices, apply_state)
