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
Collection of losses
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def get_loss(name):
  assoc = {
    'mixed': loss_mixed,
    'standard': loss_std,
    'inverse': loss_inverse,
    'logsumexp': loss_logsumexp,
  }

  return assoc[name]

def loss_vec(x, X, p):
  return tf.multiply(tf.reduce_sum(tf.square(X - x), axis=1), p) + 1e-4

def loss_logsumexp(x, X, p):
  v = loss_vec(x, X, p)
  dN = tf.cast(tf.shape(v)[0], tf.float32)
  return tf.reduce_logsumexp(-v) - tf.log(dN)

def loss_inverse(x, X, p):
  v = loss_vec(x, X, p)

  return -1.0 / tf.reduce_mean(1.0 / (v + 1e-4))

def loss_std(x, X, p):
  v = loss_vec(x, X, p)
  return - tf.reduce_mean(v, axis=0, name='loss_std')

def loss_mixed(x, Lx, px, scale=1.0):
  v1 = loss_vec(x, Lx, px)
  v1 /= scale
  sampler_loss = 0.
  sampler_loss += (tf.reduce_mean(1.0 / v1, name='loss_inv'))
  sampler_loss += (- tf.reduce_mean(v1, name='loss_mixed'))
  return sampler_loss
