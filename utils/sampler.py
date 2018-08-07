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
Sampling functions given dynamics and placeholders
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

TF_FLOAT = tf.float32

def propose(x, dynamics, init_v=None, aux=None,
            do_mh_step=False, log_jac=False):
    if dynamics.hmc:
        Lx, Lv, px = dynamics.forward(x, init_v=init_v, aux=aux)
        return Lx, Lv, px, [tf_accept(x, Lx, px)]
    else:
        # sample mask for forward/backward
        mask = tf.cast(tf.random_uniform((tf.shape(x)[0], 1),
                                         maxval=2, dtype=tf.int32),
                       TF_FLOAT)
        Lx1, Lv1, px1 = dynamics.forward(x, aux=aux, log_jac=log_jac)
        Lx2, Lv2, px2 = dynamics.backward(x, aux=aux, log_jac=log_jac)

        Lx = mask * Lx1 + (1 - mask) * Lx2

        Lv = None
        if init_v is not None:
            Lv = mask * Lv1 + (1 - mask) * Lv2

        px = (tf.squeeze(mask, axis=1) * px1
              + tf.squeeze(1 - mask, axis=1) * px2)

        outputs = []

        if do_mh_step:
            outputs.append(tf_accept(x, Lx, px))

        return Lx, Lv, px, outputs

def tf_accept(x, Lx, px):
    mask = (px - tf.random_uniform(tf.shape(px)) >= 0.)
    return tf.where(mask, Lx, x)

def chain_operator(init_x, dynamics, nb_steps,
                   aux=None, init_v=None, do_mh_step=False):
    if not init_v:
        init_v = tf.random_normal(tf.shape(init_x))

    def cond(latent, v, log_jac, t):
        return tf.less(t, tf.cast(nb_steps, tf.float32))

    def body(x, v, log_jac, t):
        Lx, Lv, px, _ = propose(x, dynamics, init_v=v,
                                aux=aux, log_jac=True, do_mh_step=False)
        return Lx, Lv, log_jac+px, t+1

    final_x, final_v, log_jac, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[
                init_x,
                init_v,
                tf.zeros((tf.shape(init_x)[0],)),
                tf.constant(0.),
            ]
        )

    p_accept = dynamics.p_accept(init_x, init_v,
                                 final_x, final_v,
                                 log_jac, aux=aux)

    outputs = []
    if do_mh_step:
        outputs.append(tf_accept(init_x, final_x, p_accept))

    return final_x, final_v, p_accept, outputs
