"""TODO(danilevy): DO NOT SUBMIT without one-line documentation for layers.

TODO(danilevy): DO NOT SUBMIT without a detailed description of layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.google as tf
import numpy as np

from google3.pyglib import app
from google3.pyglib import flags

class Linear(object):
  def __init__(self, in_, out_, scope='linear', factor=1.0):
    with tf.variable_scope(scope):
      initializer = tf.contrib.layers.variance_scaling_initializer(factor=factor * 2.0, mode='FAN_IN', uniform=False)
      self.W = tf.get_variable('W', shape=(in_, out_), initializer=initializer)
      self.b = tf.get_variable('b', shape=(out_,), initializer=tf.constant_initializer(0.))

  def __call__(self, x):
    return tf.add(tf.matmul(x, self.W), self.b)

class ScaledTanh(object):
  def __init__(self, in_, scope='scaled_tanh'):
    with tf.variable_scope(scope):
      self.scale = tf.exp(tf.get_variable(
          'scale',
          shape=(1, in_),
          initializer=tf.constant_initializer(0.)
      ))

  def __call__(self, x):
    return self.scale * tf.nn.tanh(x)

class ConcatLinear(object):
  def __init__(self, ins_, out_, factors=None, scope='concat_linear'):
    self.layers = []

    with tf.variable_scope(scope):
      for i, in_ in enumerate(ins_):
        if factors is None:
          factor = 1.0
        else:
          factor = factors[i]

        self.layers.append(Linear(in_, out_, scope='linear_%d' % i, factor=factor))

  def __call__(self, inputs):
    output = 0.
    for i, x in enumerate(inputs):
      output += self.layers[i](x)

    return output
