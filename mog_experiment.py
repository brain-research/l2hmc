import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import functools
import argparse
import sys
import os
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

# Model hyperparameters
X_DIM = 3
LR_INIT = 1e-3
LR_DECAY_RATE = 0.96
TEMP_INIT = 10
ANNEALING_RATE = 0.98
EPS = 0.1
SCALE = 0.1
NUM_SAMPLES = 200
TRAIN_TRAJECTORY_LENGTH = 2000
NUM_TRAINING_STEPS = 10000
LR_DECAY_STEPS = 1000
ANNEALING_STEPS = 100
LOGGING_STEPS = 50
TUNNELING_RATE_STEPS = 500
SAVE_STEPS = 2500
MEANS = np.array([[np.sqrt(2), 0.0, 0.0],
                  [0.0, np.sqrt(2), 0.0],
                  [np.sqrt(2), 0.0, 0.0]]).astype(np.float32)
SIGMA = 0.05
SMALL_PI = 2e-16

params = {
    'x_dim': X_DIM,
    'lr_init': LR_INIT,
    'temp_init': TEMP_INIT,
    'annealing_rate': ANNEALING_RATE,
    'eps': EPS,
    'scale': SCALE,
    'num_samples': NUM_SAMPLES,
    'train_trajectory_length': TRAIN_TRAJECTORY_LENGTH,
    'means': MEANS,
    'sigma': SIGMA,
    'small_pi': SMALL_PI,
    'num_training_steps': NUM_TRAINING_STEPS,
    'annealing_steps': ANNEALING_STEPS,
    'tunneling_rate_steps': TUNNELING_RATE_STEPS,
    'lr_decay_steps': LR_DECAY_STEPS,
    'save_steps': SAVE_STEPS,
    'logging_steps': LOGGING_STEPS
}

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
        if log_dir is None:
            self.log_dir, self.info_dir, self.figs_dir = self._create_log_dir()
        else:
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


        self.x_dim = params.get('x_dim', 3)
        self.lr_init = params.get('lr_init', 1e-3)
        self.lr_decay_steps = params.get('lr_decay_steps', 1000)
        self.lr_decay_rate = params.get('lr_decay_rate', 0.96)
        self.temp_init = params.get('temp_init', 10)
        self.temp = self.temp_init
        self.annealing_steps = params.get('annealing_steps', 100)
        self.annealing_rate = params.get('annealing_rate', 0.98)
        self.eps = params.get('eps', 0.1)
        self.scale = params.get('scale', 0.1)
        self.num_training_steps = params.get('num_training_steps', 2e4)
        self.num_samples = params.get('num_samples', 200)
        self.train_trajectory_length = params.get('train_trajectory_length',
                                                  2e3)
        self.sigma = params.get('sigma', 0.05)
        self.means = params.get('means', np.eye(self.x_dim))
        self.small_pi = params.get('small_pi', 2e-16)
        self.logging_steps = params.get('logging_steps', 50)
        self.tunneling_rate_steps = params.get('tunneling_rate_steps', 500)
        self.save_steps = params.get('save_steps', 2500)
        self.distribution = self._distribution(self.sigma, self.means,
                                               self.small_pi)
        self.samples = np.random.randn(self.num_samples, self.x_dim)

        self.tunneling_rates_all = {}
        self.tunneling_events_all = {}
        self.losses = []
        self.training_samples = []
        self.tunneling_info = []
        self.temp_arr = []
        self.steps_arr = []

        self.tunneling_events_all_file = (self.info_dir +
                                     'tunneling_events_all.pkl')
        self.tunneling_rates_all_file = (self.info_dir +
                                    'tunneling_rates_all.pkl')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        tf.add_to_collection('global_step', self.global_step)
        self.learning_rate = tf.train.exponential_decay(self.lr_init,
                                                        self.global_step,
                                                        self.lr_decay_steps,
                                                        self.lr_decay_rate,
                                                        staircase=True)

    def plot_tunneling_rates(self):
        #  tunneling_info_arr = np.array(self.tunneling_info)
        #  step_nums = tunneling_info_arr[:, 0]
        fig, ax = plt.subplots()
        ax.errorbar(self.steps_arr, self.tunneling_rates_avg_all,
                    yerr=self.tunneling_rates_err_all, capsize=1.5,
                    capthick=1.5, color='C0', marker='.', ls='-',
                    fillstyle='none')
        ax.set_ylabel('Tunneling rate')#, fontsize=16)
        ax.set_xlabel('Training step')#, fontsize=16)
        #ax.legend(loc='best')#, markerscale=1.5), fontsize=12)
        str1 = r"""$\mathcal{N}_1(\sqrt{2}\hat x; {{self.sigma}}), $"""
        str2 = r"""$\mathcal{N}_2(\sqrt{2}\hat y; {{self.sigma}}),$ """
        str3 = f"$T_0 =${self.temp_init};  "
        str22 = "\n"
        str4 = r"""$T \rightarrow $ """
        str5 = f"{self.annealing_rate}"
        str6 = r"""$\times T$ """
        str7 = f"every {self.annealing_steps} steps"
        #str5 = f"$ \times T$ every {annealing_steps} steps"
        #str2 = (r"$T \rightarrow $" + f"{annealing_rate}" + " \times T$ "
        #        f"every {annealing_steps} steps")
        ax.set_title(str1 + str2 + str3 + str22 + str4 + str5 + str6 + str7)
        #ax.set_ylim((-0.05, 1.))
        fig.tight_layout()
        out_file = (self.figs_dir +
                    f'3D_tunneling_rate_vs_step_num_{int(self.steps_arr[-1])}'
                    + '.png')
        print(f'Saving figure to: {out_file}')
        fig.savefig(out_file, dpi=400, bbox_inches='tight')
        return fig, ax

    def _distribution(self, sigma, means, small_pi=2e-16):
        means = np.array(means).astype(np.float32)
        cov_mtx = sigma * np.eye(self.x_dim).astype(np.float32)
        self.covs = np.array([cov_mtx, cov_mtx, cov_mtx]).astype(np.float32)
        big_pi = (1 - small_pi) / 2
        distribution = GMM(means, self.covs, [big_pi, big_pi, small_pi])
        return distribution

    def _create_log_dir(self):
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
        energy_function = self.distribution.get_energy_function()
        self.dynamics = Dynamics(self.x_dim, energy_function, T, eps,
                                 net_factory=network,
                                 use_temperature=use_temperature)

    def _create_loss(self):
        with tf.name_scope('loss'):
            self.x = tf.placeholder(tf.float32, shape=(None, self.x_dim),
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
            scale = self.scale
            self.loss += scale * (tf.reduce_mean(1.0 / v1)
                                  + tf.reduce_mean(1.0 / v2))
            self.loss += (- tf.reduce_mean(v1, name='v1')
                          - tf.reduce_mean(v2, name='v2')) / scale

    def _create_optimizer(self):
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss,
                                               global_step=self.global_step,
                                               name='train_op')

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def _create_params_file(self):
        params_file = self.info_dir + 'parameters.txt'
        with open(params_file, 'w') as f:
            f.write((f"\ninitial_temp: {self.temp_init}\n"
                     f"\nannealing_steps: {self.annealing_steps}\n"
                     f"\nannealing_factor: {self.annealing_rate}\n"
                     f"\neps: {self.eps} (initial step size; trainable)\n"
                     f"\nnum_training_steps: {self.num_training_steps}\n"
                     f"\nnum_samples: {self.num_samples}\n"
                     f"\ntrain_traj_length: {self.train_trajectory_length}\n"
                     f"\ninit_learning_rate: {self.lr_init}\n"
                     f"\nnum_decay_steps: {self.lr_decay_steps}\n"
                     f"\ndecay_rate: {self.lr_decay_rate}\n"
                     f"\nmeans:\n\n {str(self.means)}\n"
                     f"\ncovs:\n\n {str(self.covs)}\n"))
        print(f'params file written to: {params_file}')

    def _save_variables(self):
        with open(self.tunneling_events_all_file, 'wb') as f:
            pickle.dump(self.tunneling_events_all, f)
        with open(self.tunneling_rates_all_file, 'wb') as f:
            pickle.dump(self.tunneling_rates_all, f)
        np.save(self.info_dir + 'steps_array', np.array(self.steps_arr))
        np.save(self.info_dir + 'training_samples',
                np.array(self.training_samples))
        np.save(self.info_dir + 'temp_array',
                np.array(self.temp_arr))
        np.save(self.info_dir + 'tunneling_info',
                self.tunneling_info)
        np.save(self.info_dir + 'means', self.means)
        np.save(self.info_dir + 'covariances', self.covs)

    def _load_variables(self):
        self.temp_arr = list(np.load(self.info_dir + 'temp_array.npy'))
        self.training_samples = list(np.load(self.info_dir +
                                        'training_samples.npy'))
        self.tunneling_info = list(np.load(self.info_dir +
                                      'tunneling_info.npy'))
        self.means = np.load(self.info_dir + 'means.npy')
        self.covs = np.load(self.info_dir + 'covariances.npy')
        self.temp = self.temp_arr[-1]
        self.steps_arr = list(np.load(self.info_dir + 'steps_array.npy'))

        with open(self.tunneling_rates_all_file, 'rb') as f:
            self.tunneling_rates_all = pickle.load(f)
        with open(self.tunneling_events_all_file, 'rb') as f:
            self.tunneling_events_all = pickle.load(f)

    def build_graph(self):
        """Build the graph for our model."""
        if self.log_dir is None:
            self._create_log_dir()
        #energy_function = self.distribution.get_energy_function()
        self._create_dynamics(T=10, eps=self.eps, use_temperature=True)
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
        self._create_params_file()

    def generate_trajectories(self, sess):
        _samples = self.distribution.get_samples(self.num_samples)
        _trajectories = []
        for step in range(self.train_trajectory_length):
            _trajectories.append(np.copy(_samples))
            _feed_dict = {
                self.x: _samples, self.dynamics.temperature:  1.,
            }
            _samples = sess.run(self.output[0], _feed_dict)
        return np.array(_trajectories)

    def calc_tunneling_rates(self, trajectories):
        tunneling_rate = []
        tunneling_events = []
        for i in range(trajectories.shape[1]):
            events, rate = find_tunneling_events(trajectories[:, i, :],
                                                 self.means)
            tunneling_rate.append(rate)
            tunneling_events.append(events)
        tunneling_rate_avg = np.mean(tunneling_rate)
        tunneling_rate_std = np.std(tunneling_rate)
        return (tunneling_events, tunneling_rate, tunneling_rate_avg,
                tunneling_rate_std)

    def calc_tunneling_rates_errors(self, num_blocks=100):
        tunneling_rates_all_cp = {}
        for key, val in self.tunneling_rates_all.items():
            new_key = int(key)
            tunneling_rates_all_cp[new_key] = val
        tunneling_rates_all_arr = np.array(list(tunneling_rates_all_cp.values()))
        avg_tr_vals = []
        avg_tr_errs = []
        for row in tunneling_rates_all_arr:
            avg_val = np.mean(row)
            avg_tr_vals.append(avg_val)
            data_rs = block_resampling(np.array(val), num_blocks)
            avg_tr_rs = []
            for block in data_rs:
                avg_tr_rs.append(np.mean(block))
            error = jackknife_err(y_i=avg_tr_rs, y_full=avg_val,
                                  num_blocks=num_blocks) / len(row)
            avg_tr_errs.append(error)
        return avg_tr_vals, avg_tr_errs

    def train(self, num_train_steps, config=None):
        saver = tf.train.Saver(max_to_keep=3)
        initial_step = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                #  meta_file = ckpt.model_checkpoint_path + '.meta'
                #  saver = tf.train.import_meta_graph(meta_file)
                saver.restore(sess, ckpt.model_checkpoint_path)
                self.global_step = tf.train.get_global_step()
                initial_step = sess.run(self.global_step)
                self._load_variables()

            writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            #initial_step = int(self.global_step.eval())
            #initial_step = sess.run(self.global_step)

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

                if step % self.logging_steps == 0:
                    summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
                    writer.add_summary(summary_str, global_step=step)
                    writer.flush()
                    print(f"Step: {step} / {initial_step + num_train_steps}, "
                          f"Loss: {loss_:.4e}, "
                          f"Acceptance sample: {np.mean(px_):.2f}, "
                          f"LR: {lr_:.5f}, "
                          f"temp: {self.temp:.5f}\n ")

                if step % self.annealing_steps == 0:
                    if self.temp <= 1.1:
                        continue
                    else:
                        self.temp *= self.annealing_rate

                if (step + 1) % self.tunneling_rate_steps == 0:
                    self.temp_arr.append(self.temp)
                    self.steps_arr.append(step+1)

                    trajectories = self.generate_trajectories(sess)
                    self.training_samples.append(trajectories)
                    current_info = self.calc_tunneling_rates(trajectories)
                    self.tunneling_events_all[step] = current_info[0]
                    self.tunneling_rates_all[step] = current_info[1]
                    tunneling_rate_avg = current_info[2]
                    tunneling_rate_std = current_info[3]
                    tunneling_info = [step, tunneling_rate_avg,
                                      tunneling_rate_std]
                    self.tunneling_info.append(tunneling_info)

                    avg_info = self.calc_tunneling_rates_errors()
                    self.tunneling_rates_avg_all = avg_info[0]
                    self.tunneling_rates_err_all = avg_info[1]

                    print(f"\n\t Step: {step}, "
                          f"Tunneling rate avg: {tunneling_rate_avg}, "
                          f"Tunnneling rate std: {tunneling_rate_std}\n")
                    #  import pdb
                    #  pdb.set_trace()
                    self.plot_tunneling_rates()
                    self._save_variables()

                if (step + 1) % self.save_steps == 0:
                    ckpt_file = os.path.join(self.log_dir, 'model.ckpt')
                    print(f'Saving checkpoint to: {ckpt_file}')
                    saver.save(sess, ckpt_file, global_step=step)
            writer.close()


def main(args):
    if args.log_dir:
        model = GaussianMixtureModel(params, log_dir=args.log_dir)
    else:
        model = GaussianMixtureModel(params)
    if args.num_steps:
        num_train_steps = args.num_steps
    else:
        num_train_steps = 10000

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    model.build_graph()
    model.train(num_train_steps, config=config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('L2HMC model using Mixture of Gaussians '
                     'for target distribution')
    )
    parser.add_argument("-n", "--num_steps", default=10000, type=int,
                        required=True, help="Define the number of training"
                        "steps. (Default: 10000)")
    parser.add_argument("--log_dir", type=str, required=False,
                        help="Define the log dir to use if restoring from"
                        "previous run (Default: None)")
    args = parser.parse_args()

    #if args.n_steps is  not None:
    #    num_train_steps = args.n_steps
    #else:
    #    num_train_steps = 10000

    #if args.dir is not None:
    #    log_dir = args.dir
    #else:
    #    log_dir = None
    main(args)
