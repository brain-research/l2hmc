# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Distribution object providing TF Energy function, sampling (when possible)
and numpy log-density
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections

import tensorflow as tf
import numpy as np
from scipy.stats import multivariate_normal, ortho_group

def quadratic_gaussian(x, mu, S):
    return tf.diag_part(0.5 * tf.matmul(tf.matmul(x - mu, S), tf.transpose(x -
                                                                           mu)))

def random_tilted_gaussian(dim, log_min=-2., log_max=2.):
    mu = np.zeros((dim,))
    R = ortho_group.rvs(dim)
    sigma = (np.diag(np.exp(np.log(10.) * np.random.uniform(log_min, log_max,
                                                            size=(dim,))))
             + 1e-6 * np.eye(dim))
    S = R.T.dot(sigma).dot(R)
    return Gaussian(mu, S)

class Gaussian(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        print(np.linalg.det(self.sigma), self.sigma.dtype)

        self.i_sigma = np.linalg.inv(np.copy(sigma))

    def get_energy_function(self):
        def fn(x, *args, **kwargs):
            S = tf.constant(self.i_sigma.astype('float32'))
            mu = tf.constant(self.mu.astype('float32'))

            return quadratic_gaussian(x, mu, S)

        return fn

    def get_samples(self, n):
        '''
        Sampling is broken in numpy for d > 10
        '''
        C = np.linalg.cholesky(self.sigma)
        X = np.random.randn(n, self.sigma.shape[0])
        return X.dot(C.T)

    def log_density(self, X):
        return multivariate_normal(mean=self.mu, cov=self.sigma).logpdf(X)

class TiltedGaussian(Gaussian):
    def __init__(self, dim, log_min, log_max):
        self.R = ortho_group.rvs(dim)
        self.diag = (np.diag(np.exp(np.log(10.) * np.random.uniform(log_min,
                                                                    log_max,
                                                                    size=(dim,))))
                     + 1e-8 * np.eye(dim))
        S = self.R.T.dot(self.diag).dot(self.R)
        self.dim = dim
        Gaussian.__init__(self, np.zeros((dim,)), S)

    def get_samples(self, n):
        X = np.random.randn(200, self.dim)
        X = X.dot(np.sqrt(self.diag))
        X = X.dot(self.R)
        return X

class RoughWell(object):
    def __init__(self, dim, eps, easy=False):
        self.dim = dim
        self.eps = eps
        self.easy = easy

    def get_energy_function(self):
        def fn(x, *args, **kwargs):
            n = tf.reduce_sum(tf.square(x), 1)
            if not self.easy:
                return 0.5 * n + self.eps * tf.reduce_sum(tf.cos(x / (self.eps
                                                                      *
                                                                      self.eps)),
                                                          1)
            else:
                return 0.5 * n + self.eps * tf.reduce_sum(tf.cos(x/self.eps), 1)
        return fn

    def get_samples(self, n):
        # we can approximate by a gaussian for eps small enough
        return np.random.randn(n, self.dim)


class GMM(object):
    def __init__(self, mus, sigmas, pis):
        assert len(mus) == len(sigmas)
        assert np.around(np.sum(pis), len(pis)-1) == 1.0

        self.mus = mus
        self.sigmas = sigmas
        self.pis = pis

        self.nb_mixtures = len(pis)

        self.k = mus[0].shape[0]

        self.i_sigmas = []
        self.constants = []

        for i, sigma in enumerate(sigmas):
            self.i_sigmas.append(np.linalg.inv(sigma).astype('float32'))
            det = np.sqrt((2 * np.pi) ** self.k *
                          np.linalg.det(sigma)).astype('float32')
            self.constants.append((pis[i] / det).astype('float32'))

    def get_energy_function(self):
        def fn(x):
            V = tf.concat([
                    tf.expand_dims(-quadratic_gaussian(x,
                                                       self.mus[i],
                                                       self.i_sigmas[i])
                                   + tf.log(self.constants[i]), 1)
                for i in range(self.nb_mixtures)
            ], axis=1)

            return -tf.reduce_logsumexp(V, axis=1)
        return fn

    def get_samples(self, n):
        categorical = np.random.choice(self.nb_mixtures, size=(n,), p=self.pis)
        counter_samples = collections.Counter(categorical)

        samples = []

        for k, v in counter_samples.items():
            samples.append(np.random.multivariate_normal(self.mus[k],
                                                         self.sigmas[k],
                                                         size=(v,)))

        samples = np.concatenate(samples, axis=0)

        np.random.shuffle(samples)

        return samples

    def log_density(self, X):
        return np.log(sum([
            self.pis[i] * multivariate_normal(mean=self.mus[i],
                                              cov=self.sigmas[i]).pdf(X)
            for i in range(self.nb_mixtures)
        ]))


class GaussianFunnel(object):
    def __init__(self, dim=2, clip=6.):
        self.dim = dim
        self.sigma = 2.0
        self.clip = 4 * self.sigma

    def get_energy_function(self):
        print('getting energy fn')
        def fn(x):
            v = x[:, 0]
            log_p_v = tf.square(v / self.sigma)
            s = tf.exp(v)
            sum_sq = tf.reduce_sum(tf.square(x[:, 1:]), axis=1)
            n = tf.cast(tf.shape(x)[1] - 1, tf.float32)
            E = 0.5 * (log_p_v + sum_sq / s + n * tf.log(2.0 * np.pi * s))
            s_min = tf.exp(-self.clip)
            s_max = tf.exp(self.clip)
            E_safe1 = 0.5 * (log_p_v + sum_sq / s_max + n * tf.log(2.0 * np.pi
                                                                   * s_max))
            E_safe2 = 0.5 * (log_p_v + sum_sq / s_min + n * tf.log(2.0 * np.pi
                                                                   * s_min))
            E_safe = tf.minimum(E_safe1, E_safe2)

            E_ = tf.where(tf.greater(v, self.clip), E_safe1, E)
            E_ = tf.where(tf.greater(-self.clip, v), E_safe2, E_)

            return E_
        return fn

    def get_samples(self, n):
        samples = np.zeros((n, self.dim))
        for t in range(n):
            v = self.sigma * np.random.randn()
            s = np.exp(v / 2)
            samples[t, 0] = v
            samples[t, 1:] = s * np.random.randn(self.dim-1)

        return samples

    def log_density(self, x):
        v = x[:, 0]
        log_p_v = np.square(v / self.sigma)
        s = np.exp(v)
        sum_sq = np.square(x[:, 1:]).sum(axis=1)
        n = tf.shape(x)[1] - 1
        #  return 0.5 * (log_p_v + sum_sq / s + (n / 2) * tf.log(2 * tf.pi * s))
        return 0.5 * (log_p_v + sum_sq / s + (n / 2) * tf.log(2 * np.pi * s))


def gen_ring(r=1.0, var=1.0, nb_mixtures=2):
    base_points = []
    for t in range(nb_mixtures):
        c = np.cos(2 * np.pi * t / nb_mixtures)
        s = np.sin(2 * np.pi * t / nb_mixtures)
        base_points.append(np.array([r * c, r * s]))

    v = np.array(base_points)
    sigmas = [var * np.eye(2) for t in range(nb_mixtures)]

    pis = [1. / nb_mixtures] * nb_mixtures
    pis[0] += 1-sum(pis)
    return GMM(base_points, sigmas, pis)
