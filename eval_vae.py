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
Evaluates decoder using ais
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, argparse

import tensorflow as tf
import numpy as np

from utils.layers import Sequential, Linear
from utils.distributions import Gaussian
from utils.ais import ais_estimate
from utils.func_utils import get_data, binarize

from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
parser.add_argument('--leapfrogs', default=10, type=int)
parser.add_argument('--anneal_steps', default=100, type=int)
parser.add_argument('--split', default='test', type=str)
parser.add_argument('--latent_dim', default=50, type=int)
args = parser.parse_args()

with tf.variable_scope('decoder'):
    decoder = Sequential([
        Linear(args.latent_dim, 1024, scope='decoder_1'),
        tf.nn.softplus,
        Linear(1024, 1024, scope='decoder_2'),
        tf.nn.softplus,
        Linear(1024, 784, scope='decoder_3', factor=0.01)
    ])

inp = tf.placeholder(tf.float32, shape=(None, 784))
z = tf.random_normal((tf.shape(inp)[0], args.latent_dim))

gaussian = Gaussian(np.zeros((args.latent_dim,)), np.eye(args.latent_dim))
init_energy = gaussian.get_energy_function()

def final_energy(z, aux=None):
    logits = decoder(z)
    log_posterior = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=aux, logits=logits), axis=1)
    log_prior = -0.5 * tf.reduce_sum(tf.square(z), axis=1)
    return -log_posterior - log_prior

p_x_hat = ais_estimate(init_energy, final_energy, args.anneal_steps, z, x_dim=args.latent_dim, aux=inp, leapfrogs=args.leapfrogs, step_size=0.1, num_splits=50,) #refresh=True, refreshment=0.1)

saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(save_path=args.path+'model.ckpt', sess=sess)

	_, float_x_test = get_data()
	x_test = np.load(args.split+'.npy') # Fixed binarization of MNIST
	N = x_test.shape[0]

	est_log_p = 0.
	time0 = time.time()

	num_splits = 50

	for i in xrange(0, N, num_splits):
		ais_batch = x_test[i:i+num_splits]
		print ais_batch.shape
		ais_batch = ais_batch[:, np.newaxis, :] + np.zeros([1, 20, 1]).astype('float32')
		ais_batch = np.reshape(ais_batch, [-1, 784])
		print ais_batch.shape
		if i > 0:
			print '%d / %d in %.2e seconds, est=%.2f' % (i, N, time.time() - time0, est_log_p / i)
			print fetched[0]
			time0 = time.time()

		single = x_test[i, :]
		tiled = np.tile(single, (20, 1))

		fetched = sess.run(p_x_hat, {inp: ais_batch})
		est_log_p += fetched[0]
		print fetched[1]
	print(est_log_p / N)

	with open(args.path+args.split+'_ll.txt', 'a') as f:
		f.write(str(est_log_p / N)+'\n')
