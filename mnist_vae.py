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
Train a decoder-based model using L2HMC (or HMC) as a posterior sampler
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, sys, string, os

import tensorflow as tf
import numpy as np

from utils.func_utils import accept, jacobian, autocovariance, get_log_likelihood, get_data,\
    var_from_scope, binarize, normal_kl, binarize_and_shuffle
from utils.distributions import Gaussian, GMM, GaussianFunnel, gen_ring
from utils.layers import Linear, Parallel, Sequential, Zip, ScaleTanh
from utils.dynamics import Dynamics
from utils.sampler import propose, tf_accept, chain_operator
from utils.losses import get_loss, loss_mixed

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('hparams', '', 'Comma sep list of name=value')
tf.app.flags.DEFINE_string('exp_id', '', 'exp_id')

DEFAULT_HPARAMS = tf.contrib.training.HParams(
    learning_rate=0.001,
    epoch=100,
    leapfrogs=5,
    MH=5,
    optimizer='adam',
    batch_size=512,
    latent_dim=50,
    update_sampler_every=1,
    eval_samples_every=1,
    random_lf_composition=0,
    stop_gradient=False,
    hmc=False,
    eps=0.1,
    energy_scale=0.,
)

# hardcode the loss
LOSS = 'mixed'

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
    del(hps_values['eval_samples_every'])

    train_folder = string.join(
        [
            str(k)+'='+str(hps_values[k]) 
            for k in hps_values
        ],
        ',',
    )

    logdir = 'logs/%s/%s' % (FLAGS.exp_id, train_folder)

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
    
    # Setting up sampler
    def energy(z, aux=None):
        logits = decoder(z)
        log_posterior = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=aux, logits=logits), axis=1)
        log_prior = -0.5 * tf.reduce_sum(tf.square(z), axis=1)
        return (-log_posterior - log_prior)
    energy_stop_grad = lambda z, aux=None: energy(tf.stop_gradient(z), aux=None)
    sampler_loss = 0.
    
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
                        Linear(hps.latent_dim, size1, scope='embed_1', factor=0.33),
                        Linear(hps.latent_dim, size1, scope='embed_2', factor=factor * 0.33),
                        Linear(2, size1, scope='embed_3', factor=0.33),
                        encoder_sampler,
                    ]),
                    sum,
                    tf.nn.relu,
                    Linear(size1, size2, scope='linear_1'),
                    tf.nn.relu,
                    Parallel([
                        Sequential([
                            Linear(size2, hps.latent_dim, scope='linear_s', factor=0.01), 
                            ScaleTanh(hps.latent_dim, scope='scale_s')
                        ]),
                        Linear(size2, hps.latent_dim, scope='linear_t', factor=0.01),
                        Sequential([
                            Linear(size2, hps.latent_dim, scope='linear_f', factor=0.01),
                            ScaleTanh(hps.latent_dim, scope='scale_f'),
                        ])
                    ])
                ])
            return net
        
        dynamics = Dynamics(
            hps.latent_dim, 
            energy, 
            T=hps.leapfrogs, 
            eps=hps.eps, 
            hmc=hps.hmc, 
            net_factory=net_factory, 
            eps_trainable=True, 
            use_temperature=False,
        )
        


    init_x = tf.stop_gradient(latent_q)
    init_v = tf.random_normal(tf.shape(init_x))

    for t in range(hps.MH):
        inverse_term = 0.
        other_term = 0.
        energy_loss = 0.

        if hps.stop_gradient:
            init_x = tf.stop_gradient(init_x)

        if hps.random_lf_composition > 0:
            nb_steps = tf.random_uniform((), minval=1, maxval=hps.random_lf_composition, dtype=tf.int32)

            final_x, _, px, MH = chain_operator(init_x, dynamics, nb_steps, aux=inp, do_mh_step=True)

            energy_loss = 0.

        else:
            inverse_term = 0.
            other_term = 0.

            final_x, _, px, MH = propose(init_x, dynamics, aux=inp, do_mh_step=True)
            
            #sampler_loss += 1.0 / hps.MH * loss_mixed(latent, Lx, px, scale=tf.stop_gradient(tf.exp(log_sigma)))
            
        # distance
        v = tf.square(final_x - init_x) / (tf.stop_gradient(tf.exp(2 * log_sigma)) + 1e-4)    
        v = tf.reduce_sum(v, 1) * px + 1e-4

        # energy

        energy_diff = tf.square(energy(final_x, aux=inp) - energy(init_x, aux=inp)) * px + 1e-4

        inverse_term += 1.0 / hps.MH * tf.reduce_mean(1.0 / v)
        other_term -= 1.0 / hps.MH * tf.reduce_mean(v)
        energy_loss += 1.0 / hps.MH * (tf.reduce_mean(1.0 / energy_diff) - tf.reduce_mean(energy_diff))

        init_x = MH[0]

    latent_T = init_x

    sampler_loss = inverse_term + other_term + hps.energy_scale * energy_loss

    
    logits_T = decoder(tf.stop_gradient(latent_T))
    partition = tf.constant(np.sqrt((2 * np.pi) ** hps.latent_dim), dtype=tf.float32)
    prior_probs = tf.log(partition) + \
        0.5 * tf.reduce_sum(tf.square(tf.stop_gradient(latent_T)), axis=1)
    posterior_probs = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inp, logits=logits_T), axis=1)

    likelihood = tf.reduce_mean(prior_probs+posterior_probs, axis=0)

    kl = normal_kl(mu, tf.exp(log_sigma), 0., 1.)
    bce = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inp, logits=logits), axis=1)
    elbo = tf.check_numerics(tf.reduce_mean(kl+bce), 'elbo NaN')
    
    batch_per_epoch = N / hps.batch_size

    # Setting up train ops

    global_step = tf.Variable(0., trainable=False)
    # learning_rate = tf.train.exponential_decay(
    #     hps.learning_rate, 
    #     global_step,
    #     750,
    #     0.96, 
    #     staircase=True
    # )
    
    learning_rate = tf.train.piecewise_constant(global_step, [batch_per_epoch * 500.], [1e-3, 1e-4])
    
    opt_sampler = tf.train.AdamOptimizer(learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    
    elbo_train_op = opt.minimize(elbo, var_list=var_from_scope('encoder'))
    if not hps.hmc:
        gradients, variables = zip(*opt_sampler.compute_gradients(sampler_loss, var_list=var_from_scope('sampler')))
        gradients, global_norm = tf.clip_by_global_norm(gradients, 5.0)
        sampler_train_op = opt_sampler.apply_gradients(zip(gradients, variables))
        # sampler_train_op = opt_sampler.minimize(sampler_loss, var_list=var_from_scope('sampler'), global_step=global_step)
    else:
        sampler_train_op = tf.no_op()
    decoder_train_op = opt.minimize(likelihood, var_list=var_from_scope('decoder'), global_step=global_step)

    # if not hps.hmc:
    #    tf.summary.scalar('sampler_grad_norm', global_norm)

    tf.summary.scalar('inverse_term', inverse_term)
    tf.summary.scalar('other_term', other_term)
    tf.summary.scalar('energy_loss', energy_loss)
    tf.summary.scalar('sampler_loss', sampler_loss)
    tf.summary.scalar('log_prob', likelihood)
    tf.summary.scalar('elbo', elbo)
    tf.summary.scalar('p_accept', tf.reduce_mean(px))
    
    loss_summaries = tf.summary.merge_all()

    # For sample generation
    z_eval = tf.placeholder(tf.float32, shape=(None, hps.latent_dim))
    x_eval = tf.nn.sigmoid(decoder(z_eval))

    samples_summary = tf.summary.image(
        'samples',
        tf.reshape(x_eval, (-1, 28, 28, 1)),
        64,
    )
    
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(logdir)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    counter = 0

    # For graph restore
    tf.add_to_collection('inp', inp)
    tf.add_to_collection('latent_q', latent_q)
    tf.add_to_collection('latent_T', latent_T)
    tf.add_to_collection('logits_T', logits_T)
    tf.add_to_collection('z_eval', z_eval)
    tf.add_to_collection('x_eval', x_eval)

    time0 = time.time()
    for e in range(hps.epoch):
        x_train = binarize_and_shuffle(float_x_train)
        
        for t in range(batch_per_epoch):
            start = t * hps.batch_size
            end = start + hps.batch_size
            
            batch = x_train[start:end, :]
            
            fetches = [
                elbo, sampler_loss, likelihood, loss_summaries, \
                global_step, elbo_train_op, decoder_train_op, learning_rate
            ]
                        
            if t % hps.update_sampler_every == 0:
                fetches += [sampler_train_op]
                
            fetched = sess.run(fetches, {inp: batch})
                        
            if t % 50 == 0:
                print 'Step:%d::%d/%d::ELBO: %.3e::Loss sampler: %.3e:: Log prob: %.3e:: Lr: %g:: Time: %.2e' \
                    % (fetched[4], t, batch_per_epoch, fetched[0], fetched[1], fetched[2], fetched[-2], time.time()-time0)
                time0 = time.time()

            writer.add_summary(fetched[3], global_step=counter)
            counter += 1
        if e % hps.eval_samples_every == 0:
            saver.save(sess, '%s/model.ckpt' % logdir)
            samples_summary_ = sess.run(samples_summary, {z_eval: np.random.randn(64, hps.latent_dim)})
            writer.add_summary(samples_summary_, global_step=(e / hps.eval_samples_every))

    for AS in [64, 256, 1024, 4096, 8192]:
        cmd = 'python eval_vae.py --path "%s/" --split %s --anneal_steps %d'
        print 'Train fold evaluation. AS steps: %d' % AS
        os.system(cmd % (logdir, 'train', AS))

        print 'Test fold evaluation. AS steps: %d' % AS
        os.system(cmd % (logdir, 'test', AS))

    print 'Sampler eval'
    os.system('python eval_sampler.py --path "%s"' % logdir)

if __name__ == '__main__':
    tf.app.run(main)
         
