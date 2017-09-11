# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Useful auxiliary functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
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

