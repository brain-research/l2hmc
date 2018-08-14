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
Dynamics object
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# from config import TF_FLOAT, NP_FLOAT

TF_FLOAT = tf.float32
NP_FLOAT = np.float32

def safe_exp(x, name=None):
  return tf.exp(x)
  return tf.check_numerics(tf.exp(x), message='%s is NaN' % name)

class Dynamics(object):
  def __init__(self,
               x_dim,
               energy_function,
               T=10,
               eps=0.1,
               hmc=False,
               net_factory=None,
               eps_trainable=True,
               use_temperature=False,
               log_trajectory=False):

    self.x_dim = x_dim
    self.use_temperature = use_temperature
    self.log_trajectory = log_trajectory
    self.temperature = tf.placeholder(TF_FLOAT, shape=(), name='temperature')

    if not hmc:
        alpha = tf.get_variable(
            'alpha',
            initializer=tf.log(tf.constant(eps)),
            trainable=eps_trainable,
        )
    else:
        alpha = tf.log(tf.constant(eps, dtype=TF_FLOAT))

    self.eps = safe_exp(alpha, name='alpha')
    self._fn = energy_function
    self.T = T
    self.hmc = hmc

    self._init_mask()

    # m = np.zeros((x_dim,))
    # m[np.arange(0, x_dim, 2)] = 1
    # mb = 1 - m

    # self.m = tf.constant(m, dtype=tf.float32)
    # self.mb = tf.constant(mb, dtype=tf.float32)

    # if HMC we just return all zeros
    if hmc:
      z = lambda x, *args, **kwargs: tf.zeros_like(x)
      self.XNet = lambda inp: [tf.zeros_like(inp[0]) for t in range(3)]
      self.VNet = lambda inp: [tf.zeros_like(inp[0]) for t in range(3)]
    else:
      self.XNet = net_factory(x_dim, scope='XNet', factor=2.0)
      self.VNet = net_factory(x_dim, scope='VNet', factor=1.0)
      # self.Sv, self.Tv, self.Fv = self.VNet.S, self.VNet.T, self.VNet.F
      # self.Sx, self.Tx, self.Fx = self.XNet.S, self.XNet.T, self.XNet.F


  def _init_mask(self):
    mask_per_step = []

    for t in range(self.T):
        ind = np.random.permutation(np.arange(self.x_dim))[:int(self.x_dim / 2)]
        m = np.zeros((self.x_dim,))
        m[ind] = 1
        mask_per_step.append(m)

    self.mask = tf.constant(np.stack(mask_per_step), dtype=TF_FLOAT)

  def _get_mask(self, step):
    m = tf.gather(self.mask, tf.cast(step, dtype=tf.int32))
    return m, 1.-m

  def _format_time(self, t, tile=1):
    trig_t = tf.squeeze([
        tf.cos(2 * np.pi * t / self.T),
        tf.sin(2 * np.pi * t / self.T),
    ])

    return tf.tile(tf.expand_dims(trig_t, 0), (tile, 1))

  def kinetic(self, v):
    return 0.5 * tf.reduce_sum(tf.square(v), axis=1)

  def clip_with_grad(self, u, min_u=-32., max_u=32.):
    u = u - tf.stop_gradient(tf.nn.relu(u - max_u))
    u = u + tf.stop_gradient(tf.nn.relu(min_u - u))
    return u

  def _forward_step(self, x, v, step, aux=None):
    t = self._format_time(step, tile=tf.shape(x)[0])

    grad1 = self.grad_energy(x, aux=aux)
    S1 = self.VNet([x, grad1, t, aux])

    sv1 = 0.5 * self.eps * S1[0]
    tv1 = S1[1]
    fv1 = self.eps * S1[2]

    v_h = (tf.multiply(v, safe_exp(sv1, name='sv1F'))
           + 0.5 * self.eps * (-tf.multiply(safe_exp(fv1, name='fv1F'),
                                            grad1) + tv1))

    m, mb = self._get_mask(step)

    # m, mb = self._gen_mask(x)

    X1 = self.XNet([v_h, m * x, t, aux])

    sx1 = (self.eps * X1[0])
    tx1 = X1[1]
    fx1 = self.eps * X1[2]

    y = (m * x + mb * (tf.multiply(x, safe_exp(sx1, name='sx1F'))
                       + self.eps * (tf.multiply(safe_exp(fx1, name='fx1F'),
                                                 v_h) + tx1)))

    X2 = self.XNet([v_h, mb * y, t, aux])

    sx2 = (self.eps * X2[0])
    tx2 = X2[1]
    fx2 = self.eps * X2[2]

    x_o = (mb * y + m * (tf.multiply(y, safe_exp(sx2, name='sx2F'))
                         + self.eps * (tf.multiply(safe_exp(fx2, name='fx2F'),
                                                   v_h) + tx2)))

    S2 = self.VNet([x_o, self.grad_energy(x_o, aux=aux), t, aux])
    sv2 = (0.5 * self.eps * S2[0])
    tv2 = S2[1]
    fv2 = self.eps * S2[2]

    grad2 = self.grad_energy(x_o, aux=aux)
    v_o = (tf.multiply(v_h, safe_exp(sv2, name='sv2F'))
           + 0.5 * self.eps * (-tf.multiply(safe_exp(fv2, name='fv2F'),
                                            grad2) + tv2))

    log_jac_contrib = tf.reduce_sum(sv1 + sv2 + mb * sx1 + m * sx2, axis=1)

    return x_o, v_o, log_jac_contrib

  def _backward_step(self, x_o, v_o, step, aux=None):
    t = self._format_time(step, tile=tf.shape(x_o)[0])

    grad1 = self.grad_energy(x_o, aux=aux)

    S1 = self.VNet([x_o, grad1, t, aux])

    sv2 = (-0.5 * self.eps * S1[0])
    tv2 = S1[1]
    fv2 = self.eps * S1[2]

    v_h = (tf.multiply((v_o - 0.5 * self.eps
                        * (-tf.multiply(safe_exp(fv2, name='fv2B'), grad1)
                           + tv2)), safe_exp(sv2, name='sv2B')))

    m, mb = self._get_mask(step)

    # m, mb = self._gen_mask(x_o)

    X1 = self.XNet([v_h, mb * x_o, t, aux])

    sx2 = (-self.eps * X1[0])
    tx2 = X1[1]
    fx2 = self.eps * X1[2]

    y = (mb * x_o + m * tf.multiply(safe_exp(sx2, name='sx2B'),
                                    (x_o - self.eps * (tf.multiply(
                                        safe_exp(fx2, name='fx2B'), v_h)
                                        + tx2))))

    X2 = self.XNet([v_h, m * y, t, aux])

    sx1 = (-self.eps * X2[0])
    tx1 = X2[1]
    fx1 = self.eps * X2[2]

    x = m * y + mb * tf.multiply(safe_exp(sx1, name='sx1B'),
                                 (y - self.eps * (tf.multiply(
                                     safe_exp(fx1, name='fx1B'), v_h) + tx1)
                                 ))

    grad2 = self.grad_energy(x, aux=aux)
    S2 = self.VNet([x, grad2, t, aux])

    sv1 = (-0.5 * self.eps * S2[0])
    tv1 = S2[1]
    fv1 = self.eps * S2[2]

    v = tf.multiply(safe_exp(sv1, name='sv1B'),
                    (v_h - 0.5 * self.eps * (-tf.multiply(
                        safe_exp(fv1, name='fv1B'), grad2) + tv1)
                    ))

    return x, v, tf.reduce_sum(sv1 + sv2 + mb * sx1 + m * sx2, axis=1)

  def energy(self, x, aux=None):
    if self.use_temperature:
      T = self.temperature
    else:
      T = tf.constant(1.0, dtype=TF_FLOAT)

    if aux is not None:
      return self._fn(x, aux=aux) / T
    else:
      return self._fn(x) / T

  def hamiltonian(self, x, v, aux=None):
    return self.energy(x, aux=aux) + self.kinetic(v)

  def grad_energy(self, x, aux=None):
    return tf.gradients(self.energy(x, aux=aux), x)[0]

  def _gen_mask(self, x):
    dX = x.get_shape().as_list()[1]
    b = np.zeros(self.x_dim)
    for i in range(self.x_dim):
      if i % 2 == 0:
        b[i] = 1
    b = b.astype('bool')
    nb = np.logical_not(b)

    return b.astype(NP_FLOAT), nb.astype(NP_FLOAT)
#
#   def forward(self, x, init_v=None):
#     if init_v is None:
#       v = tf.random_normal(tf.shape(x))
#     else:
#       v = init_v
#
#     dN = tf.shape(x)[0]
#     j = tf.zeros((dN,))
#     curr_x, curr_v = x, v
#     for t in range(self.T):
#       curr_x, curr_v, log_j = self._forward_step(curr_x, curr_v, t)
#       j += log_j
#
#     return curr_x, curr_v, self.p_accept(x, v, curr_x, curr_v, j)

  def forward(self, x, init_v=None, aux=None, log_path=False, log_jac=False):
    if init_v is None:
      v = tf.random_normal(tf.shape(x))
    else:
      v = init_v

    dN = tf.shape(x)[0]
    t = tf.constant(0., dtype=TF_FLOAT)
    j = tf.zeros((dN,))

    def body(x, v, t, j):
      new_x, new_v, log_j = self._forward_step(x, v, t, aux=aux)
      return new_x, new_v, t+1, j+log_j

    def cond(x, v, t, j):
      return tf.less(t, self.T)

    X, V, t, log_jac_ = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=[x, v, t, j]
    )

    if log_jac:
      return X, V, log_jac_

    return X, V, self.p_accept(x, v, X, V, log_jac_, aux=aux)

  def backward(self, x, init_v=None, aux=None, log_jac=False):
    if init_v is None:
      v = tf.random_normal(tf.shape(x))
    else:
      v = init_v

    dN = tf.shape(x)[0]
    t = tf.constant(0., name='step_backward', dtype=TF_FLOAT)
    j = tf.zeros((dN,), name='acc_jac_backward')

    def body(x, v, t, j):
      new_x, new_v, log_j = self._backward_step(x, v, self.T - t - 1, aux=aux)
      return new_x, new_v, t+1, j+log_j

    def cond(x, v, t, j):
      return tf.less(t, self.T)

    X, V, t, log_jac_ = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=[x, v, t, j]
    )

    if log_jac:
      return X, V, log_jac_

    return X, V, self.p_accept(x, v, X, V, log_jac_, aux=aux)

  def p_accept(self, x0, v0, x1, v1, log_jac, aux=None):
    e_new = self.hamiltonian(x1, v1, aux=aux)
    e_old = self.hamiltonian(x0, v0, aux=aux)

    v = e_old - e_new + log_jac
    p = tf.exp(tf.minimum(v, 0.0))

    return tf.where(tf.is_finite(p), p, tf.zeros_like(p))
