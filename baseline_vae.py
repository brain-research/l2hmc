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
VAE baseline following Kingma et al. 2013
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, sys, string

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.func_utils import accept, jacobian, autocovariance, get_log_likelihood, get_data, binarize, normal_kl
from utils.distributions import Gaussian, GMM, GaussianFunnel, gen_ring
from utils.layers import Linear, Parallel, Sequential, Zip, ScaleTanh
from utils.dynamics import Dynamics

from tensorflow.examples.tutorials.mnist import input_data

def get_data():
    mnist = input_data.read_data_sets("MNIST_data/", validation_size=0)
    train_data = mnist.train.next_batch(60000, shuffle=False)[0]
    test_data = mnist.test.next_batch(10000, shuffle=False)[0]
    return train_data, test_data

def binarize_and_shuffle(x):
    N = x.shape[0]

    float_x_train = x[np.random.permutation(N), :]

    x_train = binarize(float_x_train)
    return x_train

def var_from_scope(scope_name):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)

def loss_func(x, Lx, px):
    v1 = tf.reduce_sum(tf.square(x - Lx), axis=1) * px + 1e-4
    scale = 1.0

    sampler_loss = 0.
    sampler_loss += scale * (tf.reduce_mean(1.0 / v1))
    sampler_loss += (- tf.reduce_mean(v1)) / scale
    return sampler_loss

def tf_accept(x, Lx, px):
    mask = (px - tf.random_uniform(tf.shape(px)) >= 0.)
    return tf.where(mask, Lx, x)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('hparams', '', 'Comma sep list of name=value')

DEFAULT_HPARAMS = tf.contrib.training.HParams(
    learning_rate=0.001,
    epoch=300,
    optimizer='adam',
    batch_size=512,
    latent_dim=50,
    eval_samples_every=5,
)

OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
    'nesterov': tf.train.MomentumOptimizer,
    'sgd': tf.train.GradientDescentOptimizer,
}

def main(_):
    hps = DEFAULT_HPARAMS
    print(FLAGS.hparams)
    hps.parse(FLAGS.hparams)

    # hack for logdir
    hps_values = hps.values()
    del(hps_values['epoch'])

    train_folder = string.join(
        [
            str(k)+'='+str(hps_values[k]) 
            for k in hps_values
        ],
        ',',
    )

    logdir = 'logs/baseline/%s' % train_folder

    print('Saving logs to %s' % logdir)

    float_x_train, float_x_test = get_data()
    N = float_x_train.shape[0]
    
    with tf.variable_scope('encoder'):
        encoder = Sequential([
            Linear(784, 1024, scope='encoder_1'),
            tf.nn.softplus,
            Linear(1024, 1024, scope='encoder_2'),
            tf.nn.softplus,
            Parallel([
                Linear(1024, hps.latent_dim, scope='encoder_mean'),
                Linear(1024, hps.latent_dim, scope='encoder_std'),
            ])
        ])

    with tf.variable_scope('decoder'):
        decoder = Sequential([
            Linear(hps.latent_dim, 1024, scope='decoder_1'),
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

    kl = normal_kl(mu, tf.exp(log_sigma), 0., 1.)
    bce = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inp, logits=logits), axis=1)
    elbo = tf.reduce_mean(kl+bce)
    
    opt = tf.train.AdamOptimizer(hps.learning_rate)

    tf.summary.scalar('elbo', elbo)

    loss_summaries = tf.summary.merge_all()

    elbo_train_op = opt.minimize(elbo)
    
    z_eval = tf.random_normal((64, 50))
    x_eval = tf.nn.sigmoid(decoder(z_eval))

    samples_summary = tf.summary.image(
        'samples',
        tf.reshape(x_eval, (-1, 28, 28, 1)),
        64,
    )

    time0 = time.time()
    
    batch_per_epoch = N / hps.batch_size
    
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(logdir)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    counter = 0

    for e in range(hps.epoch):
        x_train = binarize_and_shuffle(float_x_train)
        
        for t in range(batch_per_epoch):
            start = t * hps.batch_size
            end = start + hps.batch_size
            
            batch = x_train[start:end, :]
            
            fetches = [
                elbo, loss_summaries, elbo_train_op
            ]
                
            fetched = sess.run(fetches, {inp: batch})
            
            if t % 50 == 0:
                print '%d/%d::ELBO: %.2e::Time: %.2e' \
                    % (t, batch_per_epoch, fetched[0], time.time()-time0)
                time0 = time.time()

            writer.add_summary(fetched[1], global_step=counter)
            counter += 1
            
        if e % hps.eval_samples_every == 0:
            saver.save(sess, '%s/model.ckpt' % logdir)
            samples_summary_ = sess.run(samples_summary)
            writer.add_summary(samples_summary_, global_step=(e / hps.eval_samples_every))

if __name__ == '__main__':
    tf.app.run(main)
