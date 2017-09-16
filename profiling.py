import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

from utils.func_utils import accept, jacobian, autocovariance, get_log_likelihood, get_data, binarize, normal_kl
from utils.distributions import Gaussian, GMM, GaussianFunnel, gen_ring
from utils.layers import Linear, Parallel, Sequential
from utils.dynamics import Dynamics

with tf.device('/gpu:0'):
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

    def get_data():
        mnist = input_data.read_data_sets("MNIST_data/", validation_size=0)
        train_data = mnist.train.next_batch(60000, shuffle=False)[0]
        test_data = mnist.test.next_batch(10000, shuffle=False)[0]
        return train_data, test_data

    float_x_train, float_x_test = get_data()

    N = float_x_train.shape[0]
    D = float_x_train.shape[1]

    float_x_train = float_x_train[np.random.permutation(N), :]
    float_x_test = float_x_test[np.random.permutation(float_x_test.shape[0]), :]

    x_train = binarize(float_x_train)
    x_test = binarize(float_x_test)

    def var_from_scope(scope_name):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)

    inp = tf.placeholder(tf.float32, shape=(None, 784))

    mu, log_sigma = encoder(inp)

    noise = tf.random_normal(tf.shape(mu))

    latent_q = mu + noise * tf.exp(log_sigma)

    logits = decoder(latent_q)

    kl = normal_kl(mu, tf.exp(log_sigma), 0., 1.)
    bce = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inp, logits=logits), axis=1)
    elbo = tf.reduce_mean(kl+bce)
    opt = tf.train.AdamOptimizer(0.001)
    elbo_train_op = opt.minimize(elbo, var_list=var_from_scope('encoder'))

    def energy(z, aux=None):
        logits = decoder(z)
        log_posterior = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=aux, logits=logits), axis=1)
        log_prior = -0.5 * tf.reduce_sum(tf.square(z), axis=1)
        return -log_posterior - log_prior

    with tf.variable_scope('sampler'):
        x_dim = 50

        size1 = 200
        size2 = 200

        encoder_sampler = Sequential([
            Linear(784, 512, scope='encoder_1'),
            tf.nn.softplus,
            Linear(512, 512, scope='encoder_2'),
            tf.nn.softplus,
            Linear(512, size1, scope='encoder_3'),
        ])

        class Network(object):
            def __init__(self, x_dim, scope='Network', factor=1.0):
                with tf.variable_scope(scope):
                    self.embed_1 = Linear(x_dim, size1, scope='embed_1', factor=1.0 / 3)
                    self.embed_2 = Linear(x_dim, size1, scope='embed_2', factor=factor * 1.0 / 3)
                    self.embed_3 = Linear(2, size1, scope='embed_3', factor=1.0 / 3)

                    self.linear_1 = Linear(size1, size2, scope='linear_1')

                    self.scaling_S = tf.exp(tf.get_variable('scale_s', shape=(1, x_dim), initializer=tf.constant_initializer(0.)))
                    self.scaling_F = tf.exp(tf.get_variable('scale_f', shape=(1, x_dim), initializer=tf.constant_initializer(0.)))      
                    self.linear_s = Linear(size2, x_dim, scope='linear_s', factor=0.001)
                    self.linear_t = Linear(size2, x_dim, scope='linear_t', factor=0.001)
                    self.linear_f = Linear(size2, x_dim, scope='linear_f', factor=0.001)

            def hidden(self, x, v, t, aux=None):
                z1 = self.embed_1(x)
                z2 = self.embed_2(v)
                z3 = self.embed_3(t)
                z4 = encoder_sampler(aux)

                h1 = tf.nn.relu(z1 + z2 + z3 + z4)

                h2 = tf.nn.relu(self.linear_1(h1))
                # h3 = tf.nn.relu(self.linear_2(h2))
                return tf.nn.relu(h2)

            def __call__(self, inp):
                x, v, t, aux = inp
                h = self.hidden(x, v, t, aux=aux)
                S = self.scaling_S * tf.nn.tanh(self.linear_s(h))
                F = self.scaling_F * tf.nn.tanh(self.linear_f(h))
                T = self.linear_t(h)
                return S, T, F

            def S(self, x, v, t, aux=None):
                h = self.hidden(x, v, t, aux=aux)
                return self.scaling_S * tf.nn.tanh(self.linear_s(h))

            def T(self, x, v, t, aux=None):
                h = self.hidden(x, v, t, aux=aux)
                return self.linear_t(h)

            def F(self, x, v, t, aux=None):
                h = self.hidden(x, v, t, aux=aux)
                return self.scaling_F * tf.nn.tanh(self.linear_f(h))

        def net_factory(x_dim, scope, factor):
                return Network(x_dim, scope=scope, factor=factor)

        dynamics = Dynamics(x_dim, energy, T=5, eps=0.1, hmc=False, net_factory=net_factory, eps_trainable=True, use_temperature=False)

    def inverse_boolean_mask(x_true, x_false, mask):
        n = tf.shape(mask)[0]
        ind = tf.dynamic_partition(tf.range(n), mask, 2)
        return tf.dynamic_stitch(data=[x_true, x_false], indices=ind)

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

    sampler_loss = 0.

    latent = latent_q
    hmc_steps = 4
    for t in range(hmc_steps):
        #mask = tf.random_uniform((tf.shape(latent)[0],), maxval=2, dtype=tf.int32)
        #x1, x2 = tf.dynamic_partition(latent, mask, 2)
        #aux1, aux2 = tf.dynamic_partition(inp, mask, 2)
        latent = tf.stop_gradient(latent)
        Lx, _, px = dynamics.forward(latent, aux=inp)
        # Lx2, _, px2 = dynamics.backward(x2, aux=aux2)

        # Lx = inverse_boolean_mask(Lx1, Lx2, mask)
        # px = inverse_boolean_mask(px1, px2, mask)

        sampler_loss += 1.0 / hmc_steps * loss_func(latent, Lx, px)

        latent = tf_accept(latent, Lx, px)

    latent_T = latent

    lr = tf.placeholder(tf.float32, shape=())
    global_step = tf.Variable(0., name='global_step', trainable=False)

    learning_rate = tf.train.exponential_decay(lr, global_step,
                                               250, 0.96, staircase=True)
    sampler_train_op = opt.minimize(sampler_loss, var_list=var_from_scope('sampler'))

    logits_T = decoder(latent_T)

    log_prob = tf.reduce_mean(
        tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inp, logits=logits_T), axis=1),
        axis=0
    )
    decoder_train_op = opt.minimize(log_prob, var_list=var_from_scope('decoder'))

# builder = tf.profiler.ProfileOptionBuilder

# tf.profiler.profile(
#     tf.get_default_graph(),
#     options=builder.trainable_variables_parameter())
    
# sess = tf.Session(config=tf.ConfigProto(
#       allow_soft_placement=True, log_device_placement=True))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

L = []

time0 = time.time()

for t in range(100):
#     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#     run_metadata = tf.RunMetadata()
    
    ind = np.random.randint(low=0, high=60000, size=(16,))
    batch = x_train[ind, :]
    # run_metadata = tf.RunMetadata()
    elbo_, sampler_loss_, log_prob_, _, _, _ = sess.run([
        elbo,
        sampler_loss,
        log_prob,
        elbo_train_op,
        sampler_train_op,
        decoder_train_op,
    ], {inp: batch, dynamics.temperature: 1.0, lr: 1e-3},)
        # options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        # run_metadata=run_metadata)
    
    # tf.profiler.profile(
    #     tf.get_default_graph(),
    #     run_meta=run_metadata,
    #     cmd='op',
    #     options=builder(builder.time_and_memory()).order_by('micros').build())
    
    L.append(elbo_+sampler_loss_+log_prob_)

    if True:
        print '%d::ELBO: %.3e::Loss sampler: %.3e:: Log prob: %.3e:: Loss: %.3e:: Time: %.2e' \
            % (t, elbo_, sampler_loss_, log_prob_, (elbo_+sampler_loss_+log_prob_), time.time()-time0)
        time0 = time.time()
#     fetched_timeline = timeline.Timeline(run_metadata.step_stats)
#     chrome_trace = fetched_timeline.generate_chrome_trace_format()
#     with open('timeline_01.json', 'w') as f:
#         f.write(chrome_trace)
    # tf.profiler.advise(tf.get_default_graph, run_meta=run_metadata)