import time

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
    epoch=1,
    leapfrogs=5,
    MH=5,
    optimizer='adam',
    batch_size=128,
    latent_dim=50,
    update_sampler_every=1,
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
    hps.parse(FLAGS.hparams)
    print(hps)
    
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
    
    # Setting up sampler
    def energy(z, aux=None):
        logits = decoder(z)
        log_posterior = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=aux, logits=logits), axis=1)
        log_prior = -0.5 * tf.reduce_sum(tf.square(z), axis=1)
        return -log_posterior - log_prior
    
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
            eps=0.1, 
            hmc=False, 
            net_factory=net_factory, 
            eps_trainable=True, 
            use_temperature=False,
        )
        


    latent = latent_q
    
    for t in range(hps.MH):
        mask = tf.cast(tf.random_uniform((tf.shape(latent)[0], 1), maxval=2, dtype=tf.int32), tf.float32)

        latent = tf.stop_gradient(latent)

        Lx1, _, px1 = dynamics.forward(latent, aux=inp)
        Lx2, _, px2 = dynamics.backward(latent, aux=inp)

        Lx = mask * Lx1 + (1 - mask) * Lx2
        px = tf.squeeze(mask, axis=1) * px1 + tf.squeeze(1 - mask, axis=1) * px2

        sampler_loss += 1.0 / hps.MH * loss_func(latent, Lx, px)

        latent = tf_accept(latent, Lx, px)

    latent_T = latent
    
    opt = tf.train.AdamOptimizer(hps.learning_rate)
    
    logits_T = decoder(tf.stop_gradient(latent_T))

    log_prob = tf.reduce_mean(
        tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inp, logits=logits_T), axis=1),
        axis=0
    )

    partition = tf.constant(np.sqrt((2 * np.pi) ** hps.latent_dim), dtype=tf.float32)
    
    log_prob += tf.reduce_mean(
        tf.log(partition) \
        + 0.5 * tf.reduce_sum(tf.square(tf.stop_gradient(latent_T)), axis=1),
        axis=0
    )

    elbo_train_op = opt.minimize(elbo, var_list=var_from_scope('encoder'))
    sampler_train_op = opt.minimize(sampler_loss, var_list=var_from_scope('sampler'))
    decoder_train_op = opt.minimize(log_prob, var_list=var_from_scope('decoder'))
    
    z_eval = tf.random_normal((64, 50))
    x_eval = tf.nn.sigmoid(decoder(z_eval))
    
    time0 = time.time()
    
    batch_per_epoch = N / hps.batch_size
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for e in range(hps.epoch):
        x_train = binarize_and_shuffle(float_x_train)
        
        for t in range(batch_per_epoch):
            start = t * hps.batch_size
            end = start + hps.batch_size
            
            batch = x_train[start:end, :]
            
            fetches = [elbo, sampler_loss, log_prob, elbo_train_op, decoder_train_op]
            
            if t % hps.update_sampler_every == 0:
                fetches += [sampler_train_op]
                
            fetched = sess.run(fetches, {inp: batch})
            
            if t % 50 == 0:
                print '%d/%d::ELBO: %.3e::Loss sampler: %.3e:: Log prob: %.3e:: Time: %.2e' \
                    % (t, batch_per_epoch, fetched[0], fetched[1], fetched[2], time.time()-time0)
                time0 = time.time()
            

if __name__ == '__main__':
    tf.app.run(main)
