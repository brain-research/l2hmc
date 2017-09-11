"""TODO(danilevy): DO NOT SUBMIT without one-line documentation for utils.

TODO(danilevy): DO NOT SUBMIT without a detailed description of utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google3.pyglib import app
from google3.pyglib import flags

import tensorflow.google as tf
import numpy as np
from scipy.stats import multivariate_normal

FLAGS = flags.FLAGS

def prout():
  return 3

def accept(x_i, x_p, p):
  assert x_i.shape == x_p.shape

  dN, dX = x_i.shape

  u = np.random.uniform(size=(dN,))

  m = (p - u >= 0).astype('int32')[:, None]

  return x_i * (1 - m) + x_p * m

def autocovariance(X, tau=1):
  """
  """

  dT, dN, dX = X.shape

  Xf = np.fft(X, axis=0)
  Cf = Xf * np.conj(Xf)
  C = np.ifft(Cf)
  auto = np.mean(C, axis=[1,2])

  return auto

#   for t in range(dT - tau):
#     x1 = X[t, :, :]
#     x2 = X[t+tau, :, :]
#
#     s += np.mean(x1 * x2)
#     # s += np.trace(x1.dot(x2.T)) / dN
#
#   return s / (dT - tau)

def jacobian(x, fx):
  return tf.transpose(tf.stack([tf.gradients(component, x)[0][0] for component in tf.unstack(fx[0])]))

def get_log_likelihood(X, gaussian):
  m = multivariate_normal(mean=gaussian.mu, cov=gaussian.sigma)
  return m.logpdf(X).mean()

