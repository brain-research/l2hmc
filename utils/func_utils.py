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

from tensorflow.examples.tutorials.mnist import input_data

def prout():
  return 3

def accept(x_i, x_p, p):
  assert x_i.shape == x_p.shape

  dN, dX = x_i.shape

  u = np.random.uniform(size=(dN,))

  m = (p - u >= 0).astype('int32')[:, None]

  return x_i * (1 - m) + x_p * m


def autocovariance(X, tau=0):
  dT, dN, dX = np.shape(X)
  s = 0.
  for t in range(dT - tau):
    x1 = X[t, :, :]
    x2 = X[t+tau, :, :]

    s += np.sum(x1 * x2) / dN

  return s / (dT - tau)

def jacobian(x, fx):
  return tf.transpose(tf.stack([tf.gradients(component, x)[0][0] for component in tf.unstack(fx[0])]))

def get_log_likelihood(X, gaussian):
  m = multivariate_normal(mean=gaussian.mu, cov=gaussian.sigma)
  return m.logpdf(X).mean()

def get_data():
  mnist = input_data.read_data_sets("MNIST_data/", validation_size=0)
  train_data = mnist.train.next_batch(60000, shuffle=False)[0]
  test_data = mnist.test.next_batch(10000, shuffle=False)[0]
  return train_data, test_data

def binarize(x):
  assert(x.max() <= 1.)
  return (np.random.random(x.shape) < x).astype(np.float32)

def tf_accept(x, Lx, px):
    mask = (px - tf.random_uniform(tf.shape(px)) >= 0.)
    return tf.where(mask, Lx, x)

def normal_kl(q_means, q_stddevs, p_means, p_stddevs):
  '''Returns the KL divergence between two normal distributions q and p.
  
  KLs are summed over the inner dimension.
  
  Args:
    `q_means`: Means of q.
    `q_stddevs`: Standard deviations of q.
    `p_means`: Means of p.
    `p_stddevs`: Standard deviations of p.
  '''

  # The log(2*pi) terms cancel, so no need to compute them.
  q_entropy = 0.5 + tf.log(q_stddevs)
  # E_q[(z - p_means)**2] = (q_means - p_means)**2 + q_stddevs**2
  q_p_cross_entropy = 0.5 * tf.square(q_stddevs / p_stddevs)
  q_p_cross_entropy += 0.5 * tf.square((q_means - p_means) / p_stddevs)
  q_p_cross_entropy += tf.log(p_stddevs)
  q_p_kl = tf.reduce_sum(-q_entropy + q_p_cross_entropy, -1)
  return q_p_kl

def binarize_and_shuffle(x):
  ''' Given a numpy array, returns a shuffled binarization

  Args:
    'x': numpy array
  '''
  N = x.shape[0]

  float_x_train = x[np.random.permutation(N), :]

  x_train = binarize(float_x_train)
  return x_train

def var_from_scope(scope_name):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)

def acl_spectrum(X, scale):
    n = X.shape[0]
    return np.array([autocovariance(X / scale, tau=t) for t in range(n-1)])

def ESS(A):
    A = A * (A > 0.05)
    return 1. / (1. + 2 * np.sum(A[1:]))
