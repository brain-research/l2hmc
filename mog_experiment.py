import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import functools
import argparse
import sys
import os
import signal
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from mpl_toolkits.mplot3d import Axes3D
from utils.func_utils import accept, jacobian, autocovariance,\
        get_log_likelihood, binarize, normal_kl, acl_spectrum, ESS
from utils.distributions import GMM
from utils.layers import Linear, Sequential, Zip, Parallel, ScaleTanh
from utils.dynamics import Dynamics
from utils.sampler import propose
from utils.notebook_utils import get_hmc_samples
from utils.logging import variable_summaries, get_run_num, make_run_dir
from utils.tunneling import distance, calc_min_distance, calc_tunneling_rate,\
        find_tunneling_events
from utils.jackknife import block_resampling, jackknife_err

# increase dimensionality until learrning slows to a crawl
# see how each of the Q, S, and T functions impacts learning rate and so on
# number of learning steps isn't limiting factor, concerned more with final
# results and eventual tunneling rate instead of intermediate tunneling rates

# look at scaling with dimensionality, look at implementing simple U2 model
# into distributions and see if any unforseen prooblems arise. 


#  go back to 2D case look aat different staarting temperaturers
#  make trajectorry length go with rroot T, go with higher initial temperatuer
# in 2D start with higher initial temp to get arorund 50% acceptance raate

#  define distance as difference in averrage plaquette
#  look at site by site difference in plaaquette (noot sum) to prevent integer
#  values that would be tnhe same across different configurrtaions
#  try to get network to be compatible with complex numbers aand eventuaally
#  complex matrrices

def distribution_arr(x_dim, n_distributions):
    assert x_dim >= n_distributions, ("n_distributions must be less than or"
                                      " equal to x_dim.")
    if x_dim == n_distributions:
        big_pi = round(1.0 / n_distributions, x_dim)
        arr = n_distributions * [big_pi]
        return np.array(arr, dtype=np.float32)
    else:
        big_pi = (1.0 / n_distributions) - 1E-16
        arr = n_distributions * [big_pi]
        small_pi = (1. - sum(arr)) / (x_dim - n_distributions)
        arr.extend((x_dim - n_distributions) * [small_pi])
        return np.array(arr, dtype=np.float32)

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def network(x_dim, scope, factor):
    with tf.variable_scope(scope):
        net = Sequential([
            Zip([
                Linear(x_dim, 10, scope='embed_1', factor=1.0 / 3),
                Linear(x_dim, 10, scope='embed_2', factor=factor * 1.0 / 3),
                Linear(2, 10, scope='embed_3', factor=1.0 / 3),
                lambda _: 0.,
            ]),
            sum,
            tf.nn.relu,
            Linear(10, 10, scope='linear_1'),
            tf.nn.relu,
            Parallel([
                Sequential([
                    Linear(10, x_dim, scope='linear_s', factor=0.001),
                    ScaleTanh(x_dim, scope='scale_s')
                ]),
                Linear(10, x_dim, scope='linear_t', factor=0.001),
                Sequential([
                    Linear(10, x_dim, scope='linear_f', factor=0.001),
                    ScaleTanh(x_dim, scope='scale_f'),
                ])
            ])
        ])
    return net

def plot_trajectory_and_distribution(samples, trajectory, x_dim=None):
    if samples.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
           alpha=0.5, marker='o', s=15, color='C0')
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color='C1', marker='o', markeredgecolor='C1', alpha=0.75,
                ls='-', lw=1., markersize=2)
    if samples.shape[1] == 2:
        fig, ax = plt.subplots()
        ax.scatter(samples[:, 0], samples[:, 1],  color='C0', alpha=0.6)
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                 color='C1', marker='o', alpha=0.8, ls='-')
    return fig, ax


class GaussianMixtureModel(object):
    """Model for training L2HMC using multiple Gaussian distributions."""
    def __init__(self, params, log_dir=None):
        """Initialize parameters and define relevant directories."""
        self._init_params(params)

        if log_dir is not None:
            if not os.path.isdir(log_dir):
                raise ValueError(f'Unable to locate {log_dir}, exiting.')
            else:
                if not log_dir.endswith('/'):
                    self.log_dir = log_dir + '/'
                else:
                    self.log_dir = log_dir
                self.info_dir = self.log_dir + 'run_info/'
                self.figs_dir = self.log_dir + 'figures/'
                if not os.path.isdir(self.info_dir):
                    os.makedirs(self.info_dir)
                if not os.path.isdir(self.figs_dir):
                    os.makedirs(self.figs_dir)
            #  self.tunneling_events_all_file = (self.info_dir +
            #                                    'tunneling_events_all.pkl')
            self.tunneling_rates_file = (self.info_dir +
                                             'tunneling_rates.pkl')
            self.params_file = self.info_dir + 'params_dict.pkl'
            self._load_variables()
        else:
            self.log_dir, self.info_dir, self.figs_dir = self._create_log_dir()

            #  self.tunneling_events_all_file = (self.info_dir +
            #                                    'tunneling_events_all.pkl')
            self.tunneling_rates_file = (self.info_dir +
                                             'tunneling_rates.pkl')
            self.params_file = self.info_dir + 'params_dict.pkl'

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        tf.add_to_collection('global_step', self.global_step)
        self.learning_rate = tf.train.exponential_decay(self.params['lr_init'],
                                                        self.global_step,
                                                        self.params['lr_decay_steps'],
                                                        self.params['lr_decay_rate'],
                                                        staircase=True)

    def _init_params(self, params):
        """Parse keys in params dictionary to be used for setting instance
        parameters."""
        self.params = {}
        self.params['x_dim'] = params.get('x_dim', 3)
        self.params['num_distributions'] = params.get('num_distributions', 2)
        self.params['lr_init'] = params.get('lr_init', 1e-3)
        self.params['lr_decay_steps'] = params.get('lr_decay_steps', 1000)
        self.params['lr_decay_rate'] = params.get('lr_decay_rate', 0.96)
        self.params['temp_init'] = params.get('temp_init', 10)
        self.params['annealing_steps'] = params.get('annealing_steps', 100)
        self.params['annealing_rate'] = params.get('annealing_rate', 0.98)
        self.params['eps'] = params.get('eps', 0.1)
        self.params['scale'] = params.get('scale', 0.1)
        self.params['num_training_steps'] = params.get('num_training_steps',
                                                       2e4)
        self.params['num_samples'] = params.get('num_samples', 200)
        ttl = params.get('train_trajectory_length', 2e3)
        self.params['train_trajectory_length'] = ttl
        self.params['sigma'] = params.get('sigma', 0.05)
        self.params['small_pi'] = params.get('small_pi', 2e-16)
        self.params['logging_steps'] = params.get('logging_steps', 100)
        self.params['tunneling_rate_steps'] = params.get('tunneling_rate_steps',
                                                         500)
        self.params['save_steps'] = params.get('save_steps', 2500)

        self.means = params.get('means', np.eye(self.params['x_dim']))
        self.distribution = self._distribution(self.params['sigma'],
                                               self.means,
                                               self.params['small_pi'])
        self.samples = np.random.randn(self.params['num_samples'],
                                       self.params['x_dim'])
        self.tunneling_rates = {}
        self.tunneling_rates_avg = []
        self.tunneling_rates_err = []
        self.losses = []
        self.tunneling_info = []
        self.temp_arr = []
        self.temp = self.params['temp_init']
        self.steps_arr = []

    def plot_tunneling_rates(self):
        """Method for plotting tunneling rates during training."""
        #  tunneling_info_arr = np.array(self.tunneling_info)
        #  step_nums = tunneling_info_arr[:, 0]
        try:
            fig, ax = plt.subplots()
            #  import pdb
            #  pdb.set_trace()
            ax.errorbar(self.steps_arr, self.tunneling_rates_avg,
                        yerr=self.tunneling_rates_err, capsize=1.5,
                        capthick=1.5, color='C0', marker='.', ls='--',
                        fillstyle='none')
            ax1 = ax.twiny()
            ax1.errorbar(self.temp_arr, self.tunneling_rates_avg,
                        yerr=self.tunneling_rates_err, capsize=1.5,
                        capthick=1.5, color='C1', marker='.', ls='-', alpha=0.8,
                        fillstyle='none')
            ax.set_ylabel('Tunneling rate')#, fontsize=16)
            ax.set_xlabel('Training step', color='C0')
            ax1.set_xlabel('Temperature', color='C1')
            ax.tick_params('x', colors='C0')
            ax1.tick_params('x', colors='C1')
            #  ax.set_xlabel('Training step')#, fontsize=16)
            #ax.legend(loc='best')#, markerscale=1.5), fontsize=12)
            str0 = (f"{self.params['num_distributions']}"
                    + f" in {self.params['x_dim']} dims; ")
            str1 = (r'$\mathcal{N}_{\hat \mu}(\sqrt{2}\hat \mu;$'
                    + r'${{{0}}}),$'.format(self.params['sigma']))
            #  str2 = (r' $\mathcal{N}_2(\sqrt{2}\hat y; $'
            #          + r'${{{0}}})$'.format(self.params['sigma']))
            ax.set_title(str0 + str1, y=1.15)
            #ax.set_ylim((-0.05, 1.))
        except ValueError:
            import pdb
            pdb.set_trace()
        fig.tight_layout()
        out_file = (self.figs_dir
                    + f'tunneling_rate_{int(self.steps_arr[-1])}.pdf')
        print(f'Saving figure to: {out_file}\n')
        fig.savefig(out_file, dpi=400, bbox_inches='tight')
        plt.close('all')
        return fig, ax

    def _distribution(self, sigma, means, small_pi=2e-16):
        """Initialize distribution using utils/distributions.py"""
        means = np.array(means).astype(np.float32)
        cov_mtx = sigma * np.eye(self.params['x_dim']).astype(np.float32)
        self.covs = np.array([cov_mtx] *
                             self.params['x_dim']).astype(np.float32)
        dist_arr = distribution_arr(self.params['x_dim'],
                                    self.params['num_distributions'])
        #  gbig_pi = (1 - small_pi) / 2
        distribution = GMM(means, self.covs, dist_arr)
        return distribution

    def _create_log_dir(self):
        """Create directory for storing information about experiment."""
        root_log_dir = './log_mog_tf/'
        log_dir = make_run_dir(root_log_dir)
        info_dir = log_dir + 'run_info/'
        figs_dir = log_dir + 'figures/'
        if not os.path.isdir(info_dir):
            os.makedirs(info_dir)
        if not os.path.isdir(figs_dir):
            os.makedirs(figs_dir)
        return log_dir, info_dir, figs_dir

    def _create_dynamics(self, T=10, eps=0.1, use_temperature=True):
        """ Create dynamics object using 'utils/dynamics.py'. """
        energy_function = self.distribution.get_energy_function()
        self.dynamics = Dynamics(self.params['x_dim'],
                                 energy_function,
                                 T,
                                 eps,
                                 net_factory=network,
                                 use_temperature=use_temperature)

    def _create_loss(self):
        """ Initialize loss and build recipe for calculating it during
        training. """
        with tf.name_scope('loss'):
            self.x = tf.placeholder(tf.float32, shape=(None,
                                                       self.params['x_dim']),
                                    name='x')
            self.z = tf.random_normal(tf.shape(self.x), name='z')
            self.Lx, _, self.px, self.output = propose(self.x, self.dynamics,
                                                       do_mh_step=True)
            self.Lz, _, self.pz, _ = propose(self.z, self.dynamics,
                                             do_mh_step=False)
            self.loss = tf.Variable(0., trainable=False, name='loss')
            v1 = ((tf.reduce_sum(tf.square(self.x - self.Lx), axis=1) * self.px)
                  + 1e-4)
            v2 = ((tf.reduce_sum(tf.square(self.z - self.Lz), axis=1) * self.pz)
                  + 1e-4)
            scale = self.params['scale']

            #  tf.assign_add(self.loss, (scale * (tf.reduce_mean(1.0 / v1)
            #                                     + tf.reduce_mean(1.0 / v2))))
            #  tf.assign_add(self.loss, (- tf.reduce_mean(v1, name='v1')
            #                            - tf.reduce_mean(v1, name='v2')) / scale)
            self.loss = self.loss + scale * (tf.reduce_mean(1.0 / v1) +
                                             tf.reduce_mean(1.0 / v2))
            self.loss = self.loss + ((- tf.reduce_mean(v1, name='v1')
                                      - tf.reduce_mean(v2, name='v2')) / scale)

    def _create_optimizer(self):
        """Initialize optimizer to be used during training."""
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss,
                                               global_step=self.global_step,
                                               name='train_op')

    def _create_summaries(self):
        """Create summary objects for logging in tensorboard."""
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def _create_params_file(self):
        """Write relevant parameters to .txt file for reference."""
        params_txt_file = self.info_dir + 'parameters.txt'
        with open(params_txt_file, 'w') as f:
            for key, value in self.params.items():
                f.write(f'\n{key}: {value}\n')
            f.write(f"\nmeans:\n\n {str(self.means)}\n"
                    f"\ncovs:\n\n {str(self.covs)}\n")

        print(f'params file written to: {params_txt_file}')

    def _save_variables(self):
        """Save current values of variables."""
        print(f"Saving parameter values to: {self.info_dir}")
        with open(self.tunneling_rates_file, 'wb') as f:
            pickle.dump(self.tunneling_rates, f)
        with open(self.params_file, 'wb') as f:
            pickle.dump(self.params, f)


        np.save(self.info_dir + 'steps_array', np.array(self.steps_arr))
        np.save(self.info_dir + 'temp_array',
                np.array(self.temp_arr))
        np.save(self.info_dir + 'tunneling_info',
                self.tunneling_info)
        np.save(self.info_dir + 'means', self.means)
        np.save(self.info_dir + 'covariances', self.covs)
        np.save(self.info_dir + 'tunneling_rates_avg',
                np.array(self.tunneling_rates_avg))
        np.save(self.info_dir + 'tunneling_rates_err',
                np.array(self.tunneling_rates_err))
        print("done!\n")

    def _load_variables(self):
        """Load variables from previously ran experiment."""
        print(f'Loading from previous parameters in from: {self.info_dir}')
        self.params = {}
        with open(self.params_file, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        self.steps_arr = list(np.load(self.info_dir + 'steps_array.npy'))
        self.temp_arr = list(np.load(self.info_dir + 'temp_array.npy'))
        self.temp = self.temp_arr[-1]

        self.means = np.load(self.info_dir + 'means.npy')
        self.covs = np.load(self.info_dir + 'covariances.npy')

        self.tunneling_info = list(np.load(self.info_dir
                                           + 'tunneling_info.npy'))
        self.tunneling_rates_avg = list(np.load(self.info_dir
                                                + 'tunneling_rates_avg.npy'))
        self.tunneling_rates_err = list(np.load(self.info_dir
                                                + 'tunneling_rates_err.npy'))


    def build_graph(self):
        """Build the graph for our model."""
        if self.log_dir is None:
            self._create_log_dir()
        #energy_function = self.distribution.get_energy_function()
        self._create_dynamics(T=10, eps=self.params['eps'],
                              use_temperature=True)
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
        self._create_params_file()

    def generate_trajectories(self, sess):
        """
        Generate trajectories using current values from L2HMC update method.
        """
        _samples = self.distribution.get_samples(self.params['num_samples'])
        _trajectories = []
        for step in range(self.params['train_trajectory_length']):
            _trajectories.append(np.copy(_samples))
            _feed_dict = {
                self.x: _samples, self.dynamics.temperature:  1.,
            }
            _samples = sess.run(self.output[0], _feed_dict)
        return np.array(_trajectories)

    def calc_tunneling_rates(self, trajectories):
        """Calculate tunneling rates from trajectories."""
        tunneling_rate = []
        for i in range(trajectories.shape[1]):
            rate = find_tunneling_events(trajectories[:, i, :], self.means,
                                         self.params['num_distributions'])
            tunneling_rate.append(rate)
        tunneling_rate_avg = np.mean(tunneling_rate)
        tunneling_rate_std = np.std(tunneling_rate)
        return (tunneling_rate, tunneling_rate_avg, tunneling_rate_std)

    def calc_tunneling_rates_errors(self, step, num_blocks=20):
        """
        Calculate tunneling rates with block jackknife resampling method for
        carrying out error analysis.
        
        Args:
            num_blocks (int):
                Number of blocks to use for block jackknife resampling.
        """
        tunneling_rates_arr = np.array(self.tunneling_rates[step])
        tunneling_rates_avg = np.mean(tunneling_rates_arr)
        tunneling_rates_avg_rs = []
        data_rs = block_resampling(np.array(tunneling_rates_arr), num_blocks)
        for block in data_rs:
            tunneling_rates_avg_rs.append(np.mean(block))
        error = jackknife_err(y_i=tunneling_rates_avg_rs,
                              y_full=tunneling_rates_avg,
                              num_blocks=num_blocks) / len(tunneling_rates_arr)
        return tunneling_rates_avg, error


    def train(self, num_train_steps, config=None):
        """Train the model."""
        saver = tf.train.Saver(max_to_keep=3)
        initial_step = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print('Restoring previous model from: '
                      f'{ckpt.model_checkpoint_path}')
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model restored.\n')
                self.global_step = tf.train.get_global_step()
                initial_step = sess.run(self.global_step)
                #  self._load_variables()

            writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            t0 = time.time()
            try:
                for step in range(initial_step, initial_step + num_train_steps):
                    feed_dict = {self.x: self.samples,
                                 self.dynamics.temperature: self.temp}

                    _, loss_, self.samples, px_, lr_, = sess.run([
                        self.train_op,
                        self.loss,
                        self.output[0],
                        self.px,
                        self.learning_rate
                    ], feed_dict=feed_dict)

                    self.losses.append(loss_)

                    if step % self.params['logging_steps'] == 0:
                        summary_str = sess.run(self.summary_op,
                                               feed_dict=feed_dict)
                        writer.add_summary(summary_str, global_step=step)
                        writer.flush()
                        print(f"Step: {step} / {initial_step + num_train_steps}, "
                              f"Loss: {loss_:.4e}, "
                              f"Acceptance sample: {np.mean(px_):.2f}, "
                              f"LR: {lr_:.5f}, "
                              f"temp: {self.temp:.5f}\n ")

                    if step % self.params['annealing_steps'] == 0:
                        tt = self.temp * self.params['annealing_rate']
                        if tt > 1:
                            self.temp = tt

                    if (step + 1) % self.params['tunneling_rate_steps'] == 0:
                        t1 = time.time()

                        self.temp_arr.append(self.temp)
                        self.steps_arr.append(step+1)

                        trajectories = self.generate_trajectories(sess)
                        tunneling_stats = self.calc_tunneling_rates(trajectories)

                        self.tunneling_rates[step] = tunneling_stats[0]
                        tunneling_rate_avg = tunneling_stats[1]
                        tunneling_rate_std = tunneling_stats[2]
                        tunneling_info = [step, tunneling_rate_avg,
                                          tunneling_rate_std]
                        self.tunneling_info.append(tunneling_info)

                        avg_info = self.calc_tunneling_rates_errors(step)
                        self.tunneling_rates_avg.append(avg_info[0])
                        self.tunneling_rates_err.append(avg_info[1])

                        print(f"\n\tStep: {step}, "
                              #  f"Tunneling rate avg: {avg_info[0][-1]}, "
                              #  f"Tunnneling rate err: {avg_info[1][-2]}\n")
                              f"Tunneling rate avg: {avg_info[0]:.4g}, "
                              f"Tunneling rate err: {avg_info[1]:.4g}, "
                              f"temp: {self.temp:.3g}")


                        #######################################################
                        #  TODO: Implement tempearture refresh if
                        #  tunneling_rate decreases.
                        #######################################################
                        new_tunneling_rate = avg_info[0]
                        prev_tunneling_rate = 0
                        if len(self.tunneling_rates_avg) > 1:
                            prev_tunneling_rate = self.tunneling_rates_avg[-2]

                        tunneling_rate_diff = (new_tunneling_rate
                                               - prev_tunneling_rate
                                               + 2 * avg_info[1])

                        #  if the tunneling rate decreased since the last time
                        #  it was calculated, restart the temperature 
                        if tunneling_rate_diff < 0:
                            # the following will revert self.temp to a value
                            # slightly smaller than the value it had previously
                            # the last time the tunneling rate was calculated
                            print("\n\tTunneling rate decreased!")
                            print("\tNew tunneling rate:"
                                  f" {new_tunneling_rate:.3g},"
                                  "Previous tunneling_rate:"
                                  f" {prev_tunneling_rate:.3g},"
                                  f"diff: {tunneling_rate_diff:.3g}\n")
                            print("\tResetting temperature...")
                            if len(self.temp_arr) > 1:
                                prev_temp = self.temp_arr[-2]
                                new_temp = (prev_temp *
                                            self.params['annealing_rate'])
                                print(f"\tCurrent temp: {self.temp:.3g}, "
                                      f"\t Previous temp: {prev_temp:.3g}, "
                                      f"\t New temp: {new_temp:.3g}\n")
                                self.temp = new_temp
                                self.temp_arr[-1] = self.temp
                        #######################################################

                        tt = time.time()
                        tunneling_time = int(tt - t1)
                        elapsed_time = int(tt - t0)
                        time_per_step100 = 100*int(tt - t0) / step

                        t_str2 = time.strftime("%H:%M:%S",
                                               time.gmtime(tunneling_time))
                        t_str = time.strftime("%H:%M:%S",
                                              time.gmtime(elapsed_time))
                        t_str3 = time.strftime("%H:%M:%S",
                                              time.gmtime(time_per_step100))

                        print(f'\tTime to calculate tunneling_rate: {t_str2}')
                        print(f'\tTime for 100 training steps: {t_str3}')
                        print(f'\tTotal time elapsed: {t_str}\n')

                        self.plot_tunneling_rates()

                    if (step + 1) % self.params['save_steps'] == 0:
                        self._save_variables()
                        ckpt_file = os.path.join(self.log_dir, 'model.ckpt')
                        print(f'Saving checkpoint to: {ckpt_file}\n')
                        saver.save(sess, ckpt_file, global_step=step)

                writer.close()

            except (KeyboardInterrupt, SystemExit):
                print("KeyboardInterrupt detected, saving current state and"
                      " exiting. ")
                #  self.plot_tunneling_rates()
                self._save_variables()
                ckpt_file = os.path.join(self.log_dir, 'model.ckpt')
                print(f'Saving checkpoint to: {ckpt_file}\n')
                saver.save(sess, ckpt_file, global_step=step)
                writer.flush()
                writer.close()


def main(args):

    X_DIM = args.dimension
    NUM_DISTRIBUTIONS = args.num_distributions

    MEANS = np.zeros((X_DIM, X_DIM), dtype=np.float32)
    CENTERS = np.sqrt(2)  # center of Gaussian
    for i in range(NUM_DISTRIBUTIONS):
        MEANS[i::NUM_DISTRIBUTIONS, i] = CENTERS


    params = {                          # default parameter values
        'x_dim': X_DIM,
        'num_distributions': NUM_DISTRIBUTIONS,
        'lr_init': 1e-3,
        'temp_init': 20,
        'annealing_rate': 0.98,
        'eps': 0.1,
        'scale': 0.1,
        'num_samples': 200,
        'train_trajectory_length': 2000,
        'means': MEANS,
        'sigma': 0.05,
        'small_pi': 2E-16,
        'num_training_steps': 20000,
        'annealing_steps': 200,
        'tunneling_rate_steps': 500,
        'lr_decay_steps': 1000,
        'save_steps': 2500,
        'logging_steps': 100
    }

    params['x_dim'] = args.dimension

    if args.step_size:
        params['eps'] = args.step_size
    if args.temp_init:
        params['temp_init'] = args.temp_init
    if args.num_steps:
        params['num_training_steps'] = args.num_steps
    if args.annealing_steps:
        params['annealing_steps'] = args.annealing_steps
    if args.annealing_rate:
        params['annealing_rate'] = args.annealing_rate

    if args.log_dir:
        model = GaussianMixtureModel(params, log_dir=args.log_dir)
    else:
        model = GaussianMixtureModel(params)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    model.build_graph()
    model.train(params['num_training_steps'], config=config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('L2HMC model using Mixture of Gaussians '
                     'for target distribution')
    )
    parser.add_argument("-d", "--dimension", type=int, required=True,
                        help="Dimensionality of distribution space.")

    parser.add_argument("-N", "--num_distributions", type=int, required=True,
                        help="Number of distributions to include for GMM model.")

    parser.add_argument("-n", "--num_steps", default=10000, type=int,
                        required=True, help="Define the number of training "
                        "steps. (Default: 10000)")

    parser.add_argument("-T", "--temp_init", default=20, type=int,
                        required=False, help="Initial temperature to use for "
                        "annealing. (Default: 20)")

    parser.add_argument("--step_size", default=0.1, type=float, required=False,
                        help="Initial step size to use in leapfrog update,"
                        "called `eps` in code. (This will be tuned for an"
                        "optimal value during" "training)")

    parser.add_argument("--annealing_steps", default=100, type=int,
                        required=False, help="Number of annealing steps."
                        "(Default: 100)")

    parser.add_argument("--annealing_rate", default=0.98, type=float,
                        required=False, help="Annealing rate. (Default: 0.98)")

    parser.add_argument("--log_dir", type=str, required=False,
                        help="Define the log dir to use if restoring from"
                        "previous run (Default: None)")

    args = parser.parse_args()

    main(args)
