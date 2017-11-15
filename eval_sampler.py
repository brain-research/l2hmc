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
Given a trained decoder and sampler returns figure of auto-covariance
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.layers import Sequential, Zip, Parallel, Linear, ScaleTanh
from utils.dynamics import Dynamics
from utils.func_utils import get_data, binarize, tf_accept, autocovariance
from utils.sampler import propose, chain_operator


parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', default='09-24', type=str)
parser.add_argument('--leapfrogs', default=5, type=int)
parser.add_argument('--latent_dim', default=50, type=int)
parser.add_argument('--MH', default=5, type=int)
parser.add_argument('--eps', default=0.01, type=float)
parser.add_argument('--path', type=str)
args = parser.parse_args()

# First load the graph and grab the mask

logdir = 'logs/%s/optimizer=adam,learning_rate=0.001,latent_dim=%d,eps=%g,MH=%d,batch_size=512,update_sampler_every=1,leapfrogs=%d,hmc=False' \
	% (args.exp_id, args.latent_dim, args.eps, args.MH, args.leapfrogs)
path = '%s/model.ckpt' % args.path

with tf.gfile.Open(path+'.meta'):
    tf.reset_default_graph()
    tf.train.import_meta_graph(path+'.meta')

mask = tf.get_default_graph().get_tensor_by_name('sampler/Const_%d:0' % 1)

with tf.Session() as sess:
    mask = sess.run(mask)

tf.reset_default_graph()

# set up model variables

with tf.variable_scope('encoder'):
    encoder = Sequential([
        Linear(784, 1024, scope='encoder_1'),
        tf.nn.softplus,
        Linear(1024, 1024, scope='encoder_2'),
        tf.nn.softplus,
        Parallel([
            Linear(1024, 50, scope='encoder_mean'),
            Linear(1024, 50, scope='encoder_std'),
        ])
    ])

with tf.variable_scope('decoder'):
    decoder = Sequential([
        Linear(50, 1024, scope='decoder_1'),
        tf.nn.softplus,
        Linear(1024, 1024, scope='decoder_2'),
        tf.nn.softplus,
        Linear(1024, 784, scope='decoder_3', factor=0.01)
    ])

# Setting up the VAE

inp = tf.placeholder(tf.float32, shape=(None, 784))

mu, log_sigma = encoder(inp)

noise = tf.random_normal(tf.shape(mu))

latent_q = mu + noise * tf.exp(log_sigma)

logits = decoder(latent_q)

# Setting up sampler
def energy(z, aux=None):
    logits = decoder(z)
    log_posterior = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=aux, logits=logits), axis=1)
    log_prior = -0.5 * tf.reduce_sum(tf.square(z), axis=1)
    return -log_posterior - log_prior


with tf.variable_scope('sampler'):
    size1 = 200
    size2 = 200

    encoder_sampler = Sequential([
        Linear(784, 512, scope='encoder_1'),
        tf.nn.softplus,
        Linear(512, 512, scope='encoder_2'),
        tf.nn.softplus,
        Linear(512, size1, scope='encoder_3'),
    ])

    def net_factory(x_dim, scope, factor):
        with tf.variable_scope(scope):
            net = Sequential([
                Zip([
                    Linear(50, size1, scope='embed_1', factor=0.33),
                    Linear(50, size1, scope='embed_2', factor=factor * 0.33),
                    Linear(2, size1, scope='embed_3', factor=0.33),
                    encoder_sampler,
                ]),
                sum,
                tf.nn.relu,
                Linear(size1, size2, scope='linear_1'),
                tf.nn.relu,
                Parallel([
                    Sequential([
                        Linear(size2, 50, scope='linear_s', factor=0.01), 
                        ScaleTanh(50, scope='scale_s')
                    ]),
                    Linear(size2, 50, scope='linear_t', factor=0.01),
                    Sequential([
                        Linear(size2, 50, scope='linear_f', factor=0.01),
                        ScaleTanh(50, scope='scale_f'),
                    ])
                ])
            ])
        return net

    dynamics = Dynamics(
        args.latent_dim, 
        energy, 
        T=args.leapfrogs, 
        eps=0.1, 
        hmc=False, 
        net_factory=net_factory, 
        eps_trainable=True, 
        use_temperature=False,
    )

dynamics.mask = tf.constant(mask, tf.float32)

# CS placeholders
z_start = tf.placeholder(tf.float32, shape=(None, 50))
# _, _, _, MH = propose(z_start, dynamics, do_mh_step=True, aux=inp)
nb_steps = tf.random_uniform((), minval=1, maxval=4, dtype=tf.int32)
_, _, _, MH = chain_operator(z_start, dynamics, nb_steps, do_mh_step=True, aux=inp)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(save_path=path, sess=sess)

# pull MNIST

train, test = get_data()
x_train = binarize(train)
x_0 = np.tile(x_train[456, :][None, :], (200, 1))

init_chain = sess.run(latent_q, {inp: x_0})

list_samples = []

samples = np.copy(init_chain)
for t in range(2000):
    list_samples.append(np.copy(samples))
    samples = sess.run(MH[0], {inp: x_0, z_start: samples})

F = np.array(list_samples)
mu = F[1000:, :, :].mean(axis=(0, 1))

for eps in np.arange(0.05, 0.2, 0.025):
    hmc_dynamics = Dynamics(
        50, 
        energy, 
        T=args.leapfrogs, 
        eps=eps, 
        hmc=True,
    )
    z_start_hmc = tf.placeholder(tf.float32, shape=(None, 50))
    _, _, _, MH_HMC = propose(z_start_hmc, hmc_dynamics, do_mh_step=True, aux=inp)
    hmc_samples = []
    samples = np.copy(init_chain)
    for t in range(2000):
        hmc_samples.append(np.copy(samples))
        samples = sess.run(MH_HMC[0], {inp: x_0, z_start_hmc: samples})
    G = np.array(hmc_samples[1000:])
    print G.shape
    plt.plot(np.abs([autocovariance(G - mu, tau=t) for t in range(199)]), label='$\epsilon=%.2f$' % eps)
plt.plot(np.abs([autocovariance(F[1000:, :, :] - mu, tau=t) for t in range(199)]), label='CS')
plt.xlabel('# MH steps')
plt.ylabel('Autocovariance')
plt.legend()

plt.savefig('%s/sampler_eval.png' % args.path)
