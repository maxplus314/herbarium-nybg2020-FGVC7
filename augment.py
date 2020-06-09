import numpy as np

import autoaugment_transforms
import autoaugment_policies


def randround(x):
  z, f = divmod(x, 1)
  return int(z + (np.random.rand() < f))


def rand_mask(l, resolution):
  dr = np.random.uniform(3/7, 1)
  dc = np.random.uniform(3/7, 1)
  rr = np.random.uniform(l / 2, l)
  rc = l * l / rr
  rc = min(rc, 1)
  rr = l * l / rc
  if np.random.randint(2):
    rr, rc = rc, rr

  ar = randround(dr * (1 - rr) * resolution[0])
  ac = randround(dc * (1 - rc) * resolution[1])
  br = randround(dr * rr * resolution[0])
  bc = randround(dc * rc * resolution[1])
  dr = np.random.randint(-br + 1, ar + 1)
  dc = np.random.randint(-bc + 1, ac + 1)

  mask = np.zeros((*resolution, 1))

  for r in range(dr, resolution[0], ar + br):
    for c in range(dc, resolution[1], ac + bc):
      mask[max(r, 0):r + br, max(c, 0):c + bc] = 1

  return mask, np.sum(mask) / np.prod(resolution)
  

def gridcutmix(xyw):
  """
    Inspired by 'GridMask Data Augmentation' [arxiv.org/abs/2001.04086]
    combined with 'CutMix: Regularization Strategy to Train Strong Classifiers
    with Localizable Features' [arxiv.org/abs/1905.04899]
  """
  x, y, w = xyw

  alpha = 1
  resolution = x.shape[1:-1]
  perm = np.random.permutation(x.shape[0])
  l = np.random.beta(alpha, alpha)
  mask, l = rand_mask(l, resolution)
  x = x + (x[perm] - x) * mask
  y = y + (y[perm] - y) * l
  w = w + (w[perm] - w) * l
  return x, y, w


class AutoAugment(object):
  def __init__(self):
    self.policies = autoaugment_policies.good_policies()
  
  def __call__(self, xyw):
    x, y, w = xyw
    x = x.copy()
    for img in x:
      policy = self.policies[np.random.choice(len(self.policies))]
      img[:] = autoaugment_transforms.apply_policy(policy, img)
    return x, y, w
