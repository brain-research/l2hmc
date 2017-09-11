"""
Tests for the Learning to MCMC project
Tests consist of:
- Forward(backward(x)) = x (one and multi-step)
- Moments are correct for standard HMC
- Moments are correct for our untrained modified HMC
- Trajectory for standard HMC corresponds to Neal's paper
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google3.pyglib import app
from google3.pyglib import flags

FLAGS = flags.FLAGS

import tensorflow.google as tf
import numpy as np

from google3.experimental.users.danilevy.l2hmc.utils.func_utils import accept, jacobian
from google3.experimental.users.danilevy.l2hmc.utils.distributions import Gaussian
from google3.experimental.users.danilevy.l2hmc.utils.layers import Linear
from google3.experimental.users.danilevy.l2hmc.utils.sampler import Sampler

NUM_SAMPLES = 200
X_DIM = 2
DISTRIBUTION = Gaussian(np.zeros((2,)), np.array([[1.0, 0.95], [0.95, 1.0]]))

size1 = 10
size2 = 10


class Network(object):
  def __init__(self, x_dim, scope='Network', factor=1.0):
    with tf.variable_scope(scope):
      self.embed_1 = Linear(x_dim, size1, scope='embed_1', factor=1.0 / 3)
      self.embed_2 = Linear(x_dim, size1, scope='embed_2', factor=factor * 1.0 / 3)
      self.embed_3 = Linear(2, size1, scope='embed_3', factor=1.0 / 3)

      self.linear_1 = Linear(size1, size2, scope='linear_1')

      self.scaling_S = tf.exp(tf.get_variable('scale_s', shape=(1, x_dim), initializer=tf.constant_initializer(0.)))
      self.scaling_F = tf.exp(tf.get_variable('scale_f', shape=(1, x_dim), initializer=tf.constant_initializer(0.)))
      self.linear_s = Linear(size2, x_dim, scope='linear_s', factor=0.1)
      self.linear_t = Linear(size2, x_dim, scope='linear_t', factor=0.1)
      self.linear_f = Linear(size2, x_dim, scope='linear_f', factor=0.1)

  def hidden(self, x, v, t, aux=None):
    z1 = self.embed_1(x)
    z2 = self.embed_2(v)
    z3 = self.embed_3(t)

    h1 = tf.nn.relu(z1 + z2 + z3)

    return tf.nn.relu(self.linear_1(h1))

  def S(self, x, v, t, aux=None):
    h = self.hidden(x, v, t)
    use_tanh = True
    if use_tanh:
      return self.scaling_S * tf.nn.tanh(self.linear_s(h))
    else:
      return self.linear_s(h)

  def T(self, x, v, t, aux=None):
    h = self.hidden(x, v, t)
    return self.linear_t(h)

  def F(self, x, v, t, aux=None):
    h = self.hidden(x, v, t)
    return self.scaling_F * tf.nn.tanh(self.linear_f(h))

def net_factory(x_dim, scope, factor):
  return Network(x_dim, scope=scope, factor=factor)

NET_FACTORY = net_factory

def rel_error(x_hat, x):
  error = np.abs(x_hat - x) / np.max(np.abs(x), np.abs(x_hat))
  return error.mean()

def moments(list_samples):
  array_samples = np.array(list_samples)
  array_samples = np.reshape(array_samples, (-1, X_DIM))

  mean = np.mean(array_samples, axis=0)
  var = np.var(array_samples, axis=0)

  true_mean = DISTRIBUTION.mu
  true_var = np.diag(DISTRIBUTION.sigma)

  print(mean, true_mean)
  print(var, true_var)

  assert np.linalg.norm(mean - true_mean) < 5e-2
  assert np.linalg.norm(var - true_var) < 1e-1

def check_moments_hmc():
  hmc_s = Sampler(X_DIM, DISTRIBUTION.get_energy_function(), T=10, eps=0.1, hmc=True, eps_trainable=False)

  x = tf.placeholder(tf.float32, shape=(NUM_SAMPLES, X_DIM))
  X, _, p = hmc_s.forward(x)

  samples = np.random.randn(NUM_SAMPLES, X_DIM)

  list_samples = []
  list_samples.append(np.copy(samples))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(500):
      proposed, p_ = sess.run([X, p], {x: samples})

      samples = accept(samples, proposed, p_)

      list_samples.append(np.copy(samples))

  moments(list_samples)

def check_radford_trajectory():
  g = Gaussian(np.zeros((2,)), np.array([[1.0, 0.95], [0.95, 1.0]]))
  
  hmc_s = Sampler(X_DIM, g.get_energy_function(), T=25, eps=0.25, hmc=True, eps_trainable=False)

  x = tf.constant(np.array([[-1.50, -1.55]]).astype('float32'))
  v = tf.constant(np.array([[-1., 1.]]).astype('float32'))

  X, V, p = hmc_s.backward(x, init_v=-v)

  expected_X = np.array([[ 0.6091, 0.0882]])
  expected_p = np.array([0.6629])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    X_, V_, p_ = sess.run([X, V, p])

    assert np.linalg.norm(X_ - expected_X) < 1e-4
    assert np.linalg.norm(expected_p - p_) < 1e-4

def check_moments():
  sampler = Sampler(
      X_DIM,
      DISTRIBUTION.get_energy_function(),
      T=10,
      eps=0.1,
      hmc=False,
      eps_trainable=True,
      net_factory=NET_FACTORY
  )

  F = tf.placeholder(tf.float32, shape=(None, 2))
  B = tf.placeholder(tf.float32, shape=(None, 2))

  F_p, _, p_f = sampler.forward(F)
  B_p, _, p_b = sampler.backward(B)

  init_samples = tf.concat([F, B], axis=0)
  proposed_samples = tf.concat([F_p, B_p], axis=0)
  p = tf.concat([p_f, p_b], axis=0)

  samples = np.random.randn(NUM_SAMPLES, X_DIM)

  list_samples = []
  list_samples.append(np.copy(samples))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for t in range(1000):
      mask = np.random.randint(low=0, high=2, size=(NUM_SAMPLES,))

      feed_dict = {
          F: samples[mask == 0],
          B: samples[mask == 1],
      }

      init_, proposed_, p_ = sess.run(
          [init_samples, proposed_samples, p],
          feed_dict
      )

      samples = accept(init_, proposed_, p_)
      list_samples.append(np.copy(samples))

  moments(list_samples)

def check_forward_backward_step():
  x = tf.random_normal((NUM_SAMPLES, X_DIM))
  v = tf.random_normal((NUM_SAMPLES, X_DIM))

  s = Sampler(
      X_DIM,
      DISTRIBUTION.get_energy_function(),
      T=10,
      eps=0.1,
      hmc=False,
      eps_trainable=True,
      net_factory=NET_FACTORY
  )

  X, V, log_jac = s._forward_step(x, v, 0)
  x2, v2, log_jac2 = s._backward_step(X, V, 0)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_, v_, log_jac_, log_jac2_, x2_, v2_ = sess.run(
        [x, v, log_jac, log_jac2, x2, v2]
    )

    assert np.linalg.norm(x_ - x2_) < 1e-5
    assert np.linalg.norm(v_ - v2_) < 1e-5
    assert np.linalg.norm(log_jac_ + log_jac2_) < 1e-5

def check_forward_backward_full():
  x = tf.random_normal((NUM_SAMPLES, X_DIM))
  v = tf.random_normal((NUM_SAMPLES, X_DIM))

  s = Sampler(
      X_DIM,
      DISTRIBUTION.get_energy_function(),
      T=10,
      eps=0.1,
      hmc=False,
      eps_trainable=True,
      net_factory=NET_FACTORY
  )
  X, V, p_acc = s.forward(x, init_v=v)
  x2, v2, p_back = s.backward(X, init_v=V)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_, v_, x2_, v2_ = sess.run([x, v, x2, v2])

    assert np.linalg.norm(x_ - x2_) < 0.0001
    assert np.linalg.norm(v_ - v2_) < 0.0001

def check_jacobian():
  x = tf.random_normal((1, X_DIM))
  v = tf.random_normal((1, X_DIM))

  def netf(x_dim, scope, factor):
    return Network(x_dim, scope=scope, factor=100 * factor)

  sampler = Sampler(
      X_DIM,
      DISTRIBUTION.get_energy_function(),
      T=10,
      eps=0.1,
      hmc=False,
      eps_trainable=True,
      net_factory=netf
  )

  X, V, log_jac = sampler._forward_step(x, v, 0)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    A, B, C, D, code_log_jac = sess.run([
        jacobian(x, X),
        jacobian(x, V),
        jacobian(v, X),
        jacobian(v, V),
        log_jac
    ])

  M = np.zeros((2 * X_DIM, 2 * X_DIM))

  M[:X_DIM, :X_DIM] = A
  M[:X_DIM, X_DIM:] = B
  M[X_DIM:, :X_DIM] = C
  M[X_DIM:, X_DIM:] = D

  real_log_jac = np.log(np.linalg.det(M))

  assert np.abs(real_log_jac - code_log_jac) < 1e-4

def check_while_loop():
  x = tf.placeholder(tf.float32, shape=(None, 2))
  v = tf.placeholder(tf.float32, shape=(None, 2))
  t = tf.placeholder(tf.float32, shape=())

  distribution = Gaussian(np.zeros((2,)), np.array([[1.0, 0.], [0., 0.1]]))

  s = Sampler(
      X_DIM,
      distribution.get_energy_function(),
      T=10,
      eps=0.3,
      hmc=False,
      eps_trainable=True,
      net_factory=NET_FACTORY
  )
  X, V, _ = s._forward_step(x, v, t)

  X2, _, _ = s.forward(x, init_v=v)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_ini, v_ini = np.random.randn(64, 2), np.random.randn(64, 2)

    x_, v_ = np.copy(x_ini), np.copy(v_ini)

    for t_ in range(10):
      x_, v_ = sess.run([X, V], {x: x_, v: v_, t: t_})

    x2_ = sess.run(X2, {x: x_ini, v: v_ini})

    print(x2_[0, :], x_[0, :])
    assert np.linalg.norm(x2_ - x_) < 1e-3

TO_RUN = [
    (check_while_loop, 'forward_step composed T times is equivalent to forward'),
    (check_jacobian, 'Log(det(jacobian)) is correct'),
    (check_forward_backward_step, 'forward(backward(x)) = x for one step'),
    (check_forward_backward_full, 'forward(backward(x)) = x for multiple steps'),
    (check_radford_trajectory, 'HMC with Neal\'s paper gives Neal\'s results'),
    (check_moments, 'Moments are correct for our method'),
    (check_moments_hmc, 'Moments are correct for HMC'),
]

def main(argv):
  tests_to_pass = len(TO_RUN)
  for i, (func, msg) in enumerate(TO_RUN):
    tf.reset_default_graph()
    func()
    print('%d / %d: %s' % (i+1, tests_to_pass, msg))

  print('All tests passed!')
  
if __name__ == '__main__':
  app.run(main)
