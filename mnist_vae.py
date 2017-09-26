import time, sys, string, os

import tensorflow as tf
import numpy as np

from utils.func_utils import accept, jacobian, autocovariance, get_log_likelihood, get_data,\
    var_from_scope, binarize, normal_kl, binarize_and_shuffle
from utils.distributions import Gaussian, GMM, GaussianFunnel, gen_ring
from utils.layers import Linear, Parallel, Sequential, Zip, ScaleTanh
from utils.dynamics import Dynamics
from utils.sampler import propose
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
    hmc=False
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

    logdir = 'logs/09-26/%s' % train_folder

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
            eps=0.1, 
            hmc=hps.hmc, 
            net_factory=net_factory, 
            eps_trainable=True, 
            use_temperature=False,
        )
        


    latent = latent_q
    
    inverse_term = 0.
    other_term = 0.
    
    only_two = True

    for t in range(hps.MH):
        latent = tf.stop_gradient(latent)
        Lx, _, px, MH = propose(latent, dynamics, aux=inp, do_mh_step=True)
        v = tf.square(Lx - latent) / (tf.stop_gradient(tf.exp(2 * log_sigma)) + 1e-4)
        
        v = tf.reduce_sum(v, 1) * px + 1e-4
        
        if only_two:
            if t < 2:
                inverse_term += 0.5 * tf.reduce_mean(1.0 / v)
                other_term -= 0.5 * tf.reduce_mean(v)
        else:
            inverse_term += 1.0 / hps.MH * tf.reduce_mean(1.0 / v)
            other_term += -1.0 / hps.MH * tf.reduce_mean(v)
        
        #sampler_loss += 1.0 / hps.MH * loss_mixed(latent, Lx, px, scale=tf.stop_gradient(tf.exp(log_sigma)))
        latent = MH[0]
        
    
    latent_T = latent
    

    sampler_loss = inverse_term + other_term
    
    opt = tf.train.AdamOptimizer(hps.learning_rate)
    
    logits_T = decoder(tf.stop_gradient(latent_T))
    partition = tf.constant(np.sqrt((2 * np.pi) ** hps.latent_dim), dtype=tf.float32)
    prior_probs = tf.log(partition) + \
        0.5 * tf.reduce_sum(tf.square(tf.stop_gradient(latent_T)), axis=1)
    posterior_probs = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inp, logits=logits_T), axis=1)

    likelihood = tf.reduce_mean(prior_probs+posterior_probs, axis=0)

    kl = normal_kl(mu, tf.exp(log_sigma), 0., 1.)
    bce = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=inp, logits=logits), axis=1)
    elbo = tf.reduce_mean(kl+bce)
    
    tf.summary.scalar('inverse_term', inverse_term)
    tf.summary.scalar('other_term', other_term)
    tf.summary.scalar('sampler_loss', sampler_loss)
    tf.summary.scalar('log_prob', likelihood)
    tf.summary.scalar('elbo', elbo)
    loss_summaries = tf.summary.merge_all()


    
    # For sample generation
    z_eval = tf.placeholder(tf.float32, shape=(None, 50))
    x_eval = tf.nn.sigmoid(decoder(z_eval))

    samples_summary = tf.summary.image(
        'samples',
        tf.reshape(x_eval, (-1, 28, 28, 1)),
        64,
    )

    
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

    opt_sampler = tf.train.AdamOptimizer(hps.learning_rate)

    elbo_train_op = opt.minimize(elbo, var_list=var_from_scope('encoder'))
    if not hps.hmc:
        sampler_train_op = opt_sampler.minimize(sampler_loss, var_list=var_from_scope('sampler'), global_step=global_step)
    else:
        sampler_train_op = tf.no_op()
    decoder_train_op = opt.minimize(likelihood, var_list=var_from_scope('decoder'))

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
                global_step, elbo_train_op, decoder_train_op
            ]
            
            if t % hps.update_sampler_every == 0:
                fetches += [sampler_train_op]
                
            fetched = sess.run(fetches, {inp: batch})
            
            if t % 50 == 0:
                print 'Step:%d::%d/%d::ELBO: %.3e::Loss sampler: %.3e:: Log prob: %.3e:: Time: %.2e' \
                    % (fetched[4], t, batch_per_epoch, fetched[0], fetched[1], fetched[2], time.time()-time0)
                time0 = time.time()

            writer.add_summary(fetched[3], global_step=counter)
            counter += 1
        if e % hps.eval_samples_every == 0:
            saver.save(sess, '%s/model.ckpt' % logdir)
            samples_summary_ = sess.run(samples_summary, {z_eval: np.random.randn(64, 50)})
            writer.add_summary(samples_summary_, global_step=(e / hps.eval_samples_every))

    cmd = 'python eval_vae.py --path "%s/" --split %s --anneal_steps 5000'

    print 'Train fold evaluation'
    os.system(cmd % (logdir, 'train'))

    print 'Test fold evaluation'
    os.system(cmd % (logdir, 'test'))

if __name__ == '__main__':
    tf.app.run(main)
