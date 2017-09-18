import time, argparse

import tensorflow as tf
import numpy as np

from utils.layers import Sequential, Linear
from utils.distributions import Gaussian
from utils.ais import ais_estimate
from utils.func_utils import get_data, binarize,

from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
args = parser.parse_args()

with tf.variable_scope('decoder'):
    decoder = Sequential([
        Linear(50, 1024, scope='decoder_1'),
        tf.nn.softplus,
        Linear(1024, 1024, scope='decoder_2'),
        tf.nn.softplus,
        Linear(1024, 784, scope='decoder_3', factor=0.01)
    ])

inp = tf.placeholder(tf.float32, shape=(None, 784))
z = tf.random_normal(tf.shape(inp))

gaussian = Gaussian(np.zeros((50,)), np.eye(50))
init_energy = gaussian.get_energy_function()

def final_energy(z, aux=None):
    logits = decoder(z)
    log_posterior = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=aux, logits=logits), axis=1)
    log_prior = -0.5 * tf.reduce_sum(tf.square(z), axis=1)
    return -log_posterior - log_prior

p_x_hat = ais_estimate(init_energy, final_energy, 1000, z, aux=inp)

saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(save_path=args.path, sess=sess)

	_, float_x_test = get_data()
	x_test = binarize(float_x_test)

	N = x_test.shape[0]

	est_log_p = 0.
	time0 = time.time()
	for i in range(N):
		if i % 100 == 0:
			print '%d / %d in %.2e seconds' % (i, N, time.time() - time0)
			time0 = time.time()

		single = x_test[i, :]
		tiled = np.tile(single, (20, 1))

		est_log_p += sess.run(p_x_hat, {inp: tiled})

	print(est_log_p)
