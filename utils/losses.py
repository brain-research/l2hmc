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
  return - tf.reduce_mean(v, axis=0)

def loss_mixed(x, Lx, px, scale=1.0):
  v1 = loss_vec(x, Lx, px)
  v1 /= scale
  sampler_loss = 0.
  sampler_loss += (tf.reduce_mean(1.0 / v1))
  sampler_loss += (- tf.reduce_mean(v1))
  return sampler_loss
