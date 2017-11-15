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
AIS implementation following Wu et al. 2016
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

from dynamics import Dynamics
from func_utils import tf_accept

def ais_estimate(
        init_energy, 
        final_energy, 
        anneal_steps, 
        initial_x,
        aux=None,
        step_size=0.5, 
        leapfrogs=25, 
        x_dim=5,
        num_splits=1,
        refresh=False,
        refreshment=0.1
    ):
    beta = tf.linspace(0., 1., anneal_steps+1)[1:]
    beta_diff = beta[1] - beta[0]
    refreshment = tf.constant(refreshment)
    def body(a, beta):
        def curr_energy(z, aux=None): 
            return (1-beta) * init_energy(z) + (beta) * final_energy(z, aux=aux)
        last_x = a[1]
        w = a[2]
        v = a[3]
        if refresh:
            refreshed_v = v * tf.sqrt(1-refreshment) + tf.random_normal(tf.shape(v)) * tf.sqrt(refreshment)
        else:
            refreshed_v = tf.random_normal(tf.shape(v))
        w = w + beta_diff * (- final_energy(last_x, aux=aux) \
            + init_energy(last_x, aux=aux))
        dynamics = Dynamics(x_dim, energy_function=curr_energy, eps=step_size, hmc=True, T=leapfrogs)
        Lx, Lv, px = dynamics.forward(last_x, aux=aux, init_v=refreshed_v)
        
        mask = (px - tf.random_uniform(tf.shape(px)) >= 0.)
        updated_x = tf.where(mask, Lx, last_x)
        updated_v = tf.where(mask, Lv, -Lv)

        return (px, updated_x, w, updated_v)

    alpha, x, w, _ = tf.scan(body, beta, 
                             (
                                 tf.zeros_like(initial_x[:, 0]),
                                 initial_x, 
                                 tf.zeros_like(initial_x[:, 0]), 
                                 tf.random_normal(tf.shape(initial_x))
                             )
                            )
    
    logmeanexp = lambda z: tf.reduce_logsumexp(z) - tf.log(tf.cast(tf.shape(z)[0], tf.float32))
    
    if num_splits == 1:
        return logmeanexp(w[-1]), tf.reduce_mean(alpha)
    
    list_w = tf.split(w[-1], num_splits, axis=0)
    return tf.reduce_sum(tf.stack(map(logmeanexp, list_w), axis=0), 0), tf.reduce_mean(alpha)
