"""TODO(danilevy): DO NOT SUBMIT without one-line documentation for sampler.

TODO(danilevy): DO NOT SUBMIT without a detailed description of sampler.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google3.pyglib import app
from google3.pyglib import flags

FLAGS = flags.FLAGS


def loss_vec(x, X, p):
  return tf.multiply(tf.reduce_sum(tf.square(X - x), axis=1), p)

def loss_logsumexp(x, X, p):
  v = loss_vec(x, X, p)
  dN = tf.cast(tf.shape(v)[0], tf.float32)

  return tf.reduce_logsumexp(-v) - tf.log(dN)

def loss_inverse(x, X, p):
  v = loss_vec(x, X, p)

  return -1.0 / tf.reduce_mean(1.0 / (v + 1e-4))

def loss_std(x, X, p):
  v = loss_vec(x, X, p)
  return - tf.reduce_mean(v, axis=0)

def loss_inv_minus_std(x, X, p, scale=0.1):
  v = loss_vec(x, X, p) + 1e-4
  loss = 0.
  loss += scale * tf.reduce_mean(1.0 / v)
  loss -= tf.reduce_mean(v) / scale
  return loss

losses = {
    'inv': loss_inverse,
    'logsumexp': loss_logsumexp,
    'std': loss_std,
    'inv_minus_std': loss_inv_minus_std,
}

class Sampler(object):
  def __init__(self, dynamics, hps):
    self.dynamics = dynamics
    self.hps = hps
    self.outputs = {}

  def propose(self, x, aux=None):
    mask = tf.random_uniform((tf.shape(x)[0],), maxval=2, dtype=tf.int32)
    x_to_forward = tf.boolean_mask(x, mask)
    x_to_backward = tf.boolean_mask(x, tf.logical_not(mask))

    x_forward_proposed, _, p_x_forward = self.dynamics.forward(x_to_forward, aux=aux)
    x_backward_proposed, _, p_x_backward = self.dynamics.backward(x_to_backward, aux=aux)

    Lx = inverse_boolean_mask(x_forward_proposed, x_backward_proposed, mask)
    px = inverse_boolean_mask(p_x_forward, p_x_backward, mask)

    return Lx, px


  def build_model(self, x, aux=None):
    Lx, px = self.propose(x, aux=aux)
    z = tf.random_normal(tf.shape(x))
    Lz, pz = self.propose(z, aux=aux)

    loss = 0.
    loss += losses[self.hps.loss](x, Lx, px)
    loss += losses[self.hps.loss](z, Lz, pz)

    self.outputs['Lx'] = Lx
    self.outputs['px'] = px
    self.outputs['loss'] = loss

    return x, Lx, px
