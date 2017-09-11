"""TODO(danilevy): DO NOT SUBMIT without one-line documentation for run.

TODO(danilevy): DO NOT SUBMIT without a detailed description of run.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google3.pyglib import app
from google3.pyglib import flags

from google3.learning.brain.python.platform import flags
from google3.learning.brain.python.platform import gfile

from google3.experimental.users.danilevy.l2hmc.utils.func_utils import accept, jacobian, autocovariance, get_log_likelihood
from google3.experimental.users.danilevy.l2hmc.utils.distributions import Gaussian, GMM, gen_ring
from google3.experimental.users.danilevy.l2hmc.utils.layers import Linear
from google3.experimental.users.danilevy.l2hmc.utils.sampler import Sampler

import tensorflow.google as tf
import numpy as np
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_string('master', 'local', 'bns')
flags.DEFINE_string('hparams', '', 'Comma sep list of name=value')
flags.DEFINE_string('train_dir', '/tmp/l2hmc/0/4', 'Training directory')

TASKS = {
    'mog': {
        'distribution': gen_ring(r=2.0, var=0.1, nb_mixtures=4),
        'eps': 0.25,
        'T': 10,
    },
    'gaussian_1': {
        'distribution': Gaussian(np.zeros(2,), np.array([[10.0, 0.], [0, 0.1]])),
        'eps': 0.4,
        'T': 10,
    },
    'gaussian_2': {
        'distribution': Gaussian(np.zeros(2,), np.array([[10.0, 0.], [0, 0.01]])),
        'eps': 0.05,
        'T': 10,
    },
}

DEFAULT_HPARAMS = tf.contrib.training.HParams(
    learning_rate=0.001,
    hidden_sizes=[10, 10],
    optimizer='adam',
    loss='inv',
    training_steps=10000,
    eval_steps=2000,
    batch_size=128,
    task='gaussian_1',
    use_temperature=False,
    start_temperature=2.0,
)

OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
    'nesterov': tf.train.MomentumOptimizer,
    'sgd': tf.train.GradientDescentOptimizer,
}

def loss_vec(x, X, p):
  return tf.multiply(tf.reduce_sum(tf.square(X - x), axis=1), p)

def loss_logsumexp(x, X, p):
  v = loss_vec(x, X, p)
  dN = tf.cast(tf.shape(v)[0], tf.float32)

  return tf.reduce_logsumexp(-v) - tf.log(dN)

def loss_inverse(x, X, p):
  v = loss_vec(x, X, p)

  return -1.0 / tf.reduce_mean(1.0 / (v + 1e-4))

def loss_std(x, X, p):
  v = loss_vec(x, X, p)
  return - tf.reduce_mean(v, axis=0)

def loss_inv_minus_std(x, X, p, scale=0.1):
  v = loss_vec(x, X, p) + 1e-4
  loss = 0.
  loss += scale * tf.reduce_mean(1.0 / v)
  loss -= tf.reduce_mean(v) / scale
  return loss

losses = {
    'inv': loss_inverse,
    'logsumexp': loss_logsumexp,
    'std': loss_std,
    'inv_minus_std': loss_inv_minus_std,
}

X_DIM = 2

def get_autocorrelation_plot(samples_1, samples_2):
  n1 = len(samples_1)
  n2 = len(samples_2)

  autocov1 = [autocovariance(np.array(samples_1), tau=t) for t in range(n1-1)]
  autocov2 = [autocovariance(np.array(samples_2), tau=t) for t in range(n2 - 1)]

  fig = plt.figure()

  plt.plot(np.abs(autocov1), label='Our Method')
  plt.plot(np.abs(autocov2), label='HMC')
  plt.legend()

  fig.canvas.draw()

  # Now we can save it to a numpy array.
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

  return data[None, :, :, :]

def get_trajectories_plot(samples_1, distribution):
  S = np.array(distribution.get_samples(1000))
  F = np.array(samples_1)

  fig = plt.figure()

  plt.scatter(S[:, 0], S[:, 1])
  plt.plot(F[:, :10, 0], F[:, :10, 1])

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

  return data[None, :, :, :]

class Network(object):
  def __init__(self, x_dim, scope='Network', factor=1.0):
    size1 = 10
    size2 = 10

    with tf.variable_scope(scope):
      self.embed_1 = Linear(x_dim, size1, scope='embed_1', factor=1.0 / 3)
      self.embed_2 = Linear(x_dim, size1, scope='embed_2', factor=factor * 1.0 / 3)
      self.embed_3 = Linear(2, size1, scope='embed_3', factor=1.0 / 3)

      self.linear_1 = Linear(size1, size2, scope='linear_1')

      self.scaling_S = tf.exp(tf.get_variable('scale_s', shape=(1, x_dim), initializer=tf.constant_initializer(0.)))
      self.scaling_F = tf.exp(tf.get_variable('scale_f', shape=(1, x_dim), initializer=tf.constant_initializer(0.)))
      self.linear_s = Linear(size2, x_dim, scope='linear_s', factor=0.1)
      self.linear_t = Linear(size2, x_dim, scope='linear_t', factor=0.1)
      self.linear_f = Linear(size2, x_dim, scope='linear_f', factor=0.1)

  def hidden(self, x, v, t):
    z1 = self.embed_1(x)
    z2 = self.embed_2(v)
    z3 = self.embed_3(t)

    h1 = tf.nn.relu(z1 + z2 + z3)

    return tf.nn.relu(self.linear_1(h1))

  def S(self, x, v, t):
    h = self.hidden(x, v, t)
    use_tanh = True
    if use_tanh:
      return self.scaling_S * tf.nn.tanh(self.linear_s(h))
    else:
      return self.linear_s(h)

  def T(self, x, v, t):
    h = self.hidden(x, v, t)
    return self.linear_t(h)

  def F(self, x, v, t):
    h = self.hidden(x, v, t)
    return self.scaling_F * tf.nn.tanh(self.linear_f(h))


def net_factory(x_dim, scope, factor):
  return Network(x_dim, scope=scope, factor=factor)

def main(_):
  hps = DEFAULT_HPARAMS
  hps.parse(FLAGS.hparams)

  g = tf.Graph()
  tf.logging.info(hps)

  with g.as_default():
    task = TASKS[hps.task]
    distribution = task['distribution']
    eps = task['eps']
    T = task['T']

    s = Sampler(
        X_DIM,
        distribution.get_energy_function(),
        T=T, eps=eps,
        hmc=False,
        eps_trainable=True,
        net_factory=net_factory,
        use_temperature=hps.use_temperature,
    )



    x_forward_proposed, _, p_x_forward = s.forward(x_to_forward)
    x_backward_proposed, _, p_x_backward = s.backward(x_to_backward)
    z_forward_proposed, _, p_z_forward = s.forward(z_to_forward)
    z_backward_proposed, _, p_z_backward = s.backward(z_to_backward)

    x_previous = tf.concat([x_to_forward, x_to_backward], axis=0)
    x_proposed = tf.concat([x_forward_proposed, x_backward_proposed], axis=0)
    p_x = tf.concat([p_x_forward, p_x_backward], axis=0)

    z_previous = tf.concat([z_to_forward, z_to_backward], axis=0)
    z_proposed = tf.concat([z_forward_proposed, z_backward_proposed], axis=0)
    p_z = tf.concat([p_z_forward, p_z_backward], axis=0)

    loss = 0.
    loss += losses[hps.loss](x_previous, x_proposed, p_x)
    loss += losses[hps.loss](z_previous, z_proposed, p_z)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('mean_acceptance_sample', tf.reduce_mean(p_x))
    tf.summary.scalar('mean_acceptance_noise', tf.reduce_mean(p_z))

    tf.summary.histogram('x_previous', x_previous)

    global_step = tf.Variable(0., name='global_step', trainable=False)

    lr = tf.constant(hps.learning_rate)
    learning_rate = tf.train.exponential_decay(lr, global_step,
                                               500, 0.96, staircase=True)

    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = OPTIMIZERS[hps.optimizer](learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    summaries = tf.summary.merge_all()

    autocorrelation_placeholder = tf.placeholder(
        tf.uint8, shape=(1, None, None, None)
    )

    autocorrelation_summary = tf.summary.image(
        'autocorrelation', autocorrelation_placeholder
    )

    trajectories_placeholder = tf.placeholder(
        tf.uint8, shape=(1, None, None, None)
    )

    trajectories_summary = tf.summary.image(
        'trajectories', trajectories_placeholder
    )

    writer = tf.summary.FileWriter(FLAGS.train_dir)

    # First get HMC samples

    with tf.variable_scope('hmc_samples'):
      hmc_sampler = Sampler(
          X_DIM, distribution.get_energy_function(), T=T, eps=eps, eps_trainable=False, hmc=True
      )

      x_hmc = tf.placeholder(tf.float32, shape=(None, X_DIM))
      X_hmc, _, p = hmc_sampler.forward(x_hmc)

    hmc_samples = []

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      samples = distribution.get_samples(hps.batch_size)

      for t in range(200):
        proposed, p_ = sess.run([X_hmc, p], {x_hmc: samples})

        hmc_samples.append(np.copy(samples))

        samples = accept(samples, proposed, p_)

    tf.logging.info('Obtained HMC samples')

    with tf.Session() as sess:
      samples = np.random.randn(hps.batch_size, X_DIM)

      loss_values = []

      sess.run(tf.global_variables_initializer())

      if hps.use_temperature:
        temperature = hps.start_temperature

      for t in range(hps.training_steps):
        if hps.use_temperature:
          temperature *= 0.99

        if not hps.use_temperature or temperature < 1.0:
          temperature = 1.0

        mask = np.random.randint(low=0, high=2, size=(hps.batch_size,))
        samples_forward = samples[mask == 0]
        samples_backward = samples[mask == 1]

        feed_dict={
            x_to_forward: samples_forward,
            x_to_backward: samples_backward,
            s.temperature: temperature,
        }

        _, loss_, summaries_, x_prev, x_prop, p_x_ = sess.run([
            train_op,
            loss,
            summaries,
            x_previous,
            x_proposed,
            p_x,
        ],
            feed_dict)

        writer.add_summary(summaries_, global_step=t)

        samples = accept(x_prev, x_prop, p_x_)
        loss_values.append(loss_)

        if t % 250 == 0:

          tf.logging.info(
              'Step: %d / %d, Loss: %.2e, Acceptance sample: %.2f'
              % (t, hps.training_steps, loss_, np.mean(p_x_))
          )

        if t % hps.eval_steps == 0:
          tf.logging.info('should eval here')
          eval_samples = []
          e_samples = distribution.get_samples(n=hps.batch_size)

          for eval_steps in range(200):
            eval_samples.append(np.copy(e_samples))

            mask = np.random.randint(low=0, high=2, size=(hps.batch_size,))
            samples_forward = e_samples[mask == 0]
            samples_backward = e_samples[mask == 1]

            feed_dict = {
                x_to_forward: samples_forward,
                x_to_backward: samples_backward,
                s.temperature: 1.0,
            }

            x_f_prop, x_b_prop, p_forward, p_backward = sess.run([
                x_forward_proposed,
                x_backward_proposed,
                p_x_forward,
                p_x_backward
            ], feed_dict)

            new_samples = np.zeros((hps.batch_size, X_DIM))
            p_accept = np.zeros((hps.batch_size,))

            new_samples[mask == 0] = x_f_prop
            new_samples[mask == 1] = x_b_prop

            p_accept[mask == 0] = p_forward
            p_accept[mask == 1] = p_backward

            e_samples = accept(e_samples, new_samples, p_accept)

          fig_autocorrelation = get_autocorrelation_plot(eval_samples, hmc_samples)
          fig_trajectories = get_trajectories_plot(eval_samples, distribution)

          autocorrelation_summary_, trajectories_summary_ = sess.run(
              [autocorrelation_summary, trajectories_summary],
              {autocorrelation_placeholder: fig_autocorrelation,
               trajectories_placeholder: fig_trajectories,
              }
          )

          writer.add_summary(autocorrelation_summary_, global_step=t)
          writer.add_summary(trajectories_summary_, global_step=t)


      tf.logging.info(hps, min(loss_values))

if __name__ == '__main__':
  tf.app.run(main)
