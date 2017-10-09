import argparse

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.layers import Sequential, Zip, Parallel, Linear, ScaleTanh
from utils.dynamics import Dynamics
from utils.func_utils import get_data, binarize, tf_accept, autocovariance
from utils.sampler import propose
from external.stats_utils import effective_n

parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', default='09-24', type=str)
parser.add_argument('--task', default='mog', type=str)
parser.add_argument('--eval_steps', default=5000, type=int)
parser.add_argument('--train_steps', default=20000, type=int)
parser.add_argument()
args = parser.parse_args()

task = TASKS[args.task]

TASKS = {
    'mog': {
        'distribution': gen_ring(r=20.0, var=1., nb_mixtures=4),
        'eps': 0.25,
        'T': 10,
        'x_dim': 2,
    },
    'gaussian_1': {
        'distribution': Gaussian(np.zeros(2,), np.array([[10.0, 0.], [0, 0.1]])),
        'eps': 0.4,
        'T': 10,
        'x_dim': 2,
    },
    'gaussian_2': {
        'distribution': Gaussian(np.zeros(2,), np.array([[10.0, 0.], [0, 0.01]])),
        'eps': 0.05,
        'T': 10,
        'x_dim': 2,
    },
}

logdir = 'toy_tasks/%s/%s' % (exp_id, args.task)

def loss_func(x, Lx, px):
	v = tf.reduce_sum(tf.square(x - Lx), 1) * px + 1e-4
	return tf.reduce_mean(1.0 / v) - tf.reduce_mean(v)

size1 = 50
size2 = 100

def net_factory(x_dim, scope, factor):
    with tf.variable_scope(scope):
        net = Sequential([
            Zip([
                Linear(task.x_dim, size1, scope='embed_1', factor=0.33),
                Linear(task.x_dim, size1, scope='embed_2', factor=factor * 0.33),
                Linear(2, size1, scope='embed_3', factor=0.33),
                lambda _, *args, **kwargs: 0.,
            ]),
            sum,
            tf.nn.relu,
            Linear(size1, size2, scope='linear_1'),
            tf.nn.relu,
            Linear(size2, size1, scope='linear_2'),
            tf.nn.relu,
            Parallel([
                Sequential([
                    Linear(size1, task.x_dim, scope='linear_s', factor=0.01), 
                    ScaleTanh(task.x_dim, scope='scale_s')
                ]),
                Linear(size1, task.x_dim, scope='linear_t', factor=0.01),
                Sequential([
                    Linear(size1, task.x_dim, scope='linear_f', factor=0.01),
                    ScaleTanh(task.x_dim, scope='scale_f'),
                ])
            ])
        ])
    return net

dynamics = Dynamics(
    task.x_dim, 
    task.distribution.get_energy_function(), 
    T=task.T, 
    eps=task.eps, 
    hmc=False, 
    eps_trainable=True, 
    net_factory=net_factory, 
    use_temperature=False
)

x = tf.placeholder(tf.float32, shape=(None, task.x_dim))
z = tf.random_normal(tf.shape(x))

Lx, _, px, MH = propose(x, dynamics, do_mh_step=True)
Lz, _, pz, _ = propose(z, dynamics, do_mh_step=False)

loss = loss_func(x, Lx, px) + loss_func(z, Lz, pz)

global_step = tf.Variable(0., trainable=False)
learning_rate = tf.train.exponential_decay(1e-3, global_step, 750, 0.98, staircase=True)

opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(loss)

with tf.Session() as sess:
	# training loop
	samples = np.random.randn(200, 2)

	for t in range(args.train_steps):
		samples, loss_, px_, _ = sess.run([MH[0], loss, px, train_op], {x: samples})    

		if t % 100 == 0:
        	print '%d/%d: Loss=%.2e, p_acc=%.2f' % (t, 20000, loss_, np.mean(px_))

    # test time eval

    init_samples = task.distribution.get_samples(200)
    samples = np.copy(init_samples)
    final_samples = []

    for t in range(args.eval_steps):
	    final_samples.append(np.copy(samples))
	    samples = sess.run(MHx[0], {x: samples})

	F = np.array(final_samples)

	mu = F.mean(axis=(0, 1))
	std = F.std(axis=(0, 1))

	all_hmc = []
	# get the HMC estimate
	eps_eval = [task.eps / 2, task.eps, 2 * task.eps]
	for eps in eps_eval:
		hmc_dynamics = Dynamics(
			task.x_dim, 
			task.distribution.get_energy_function(), 
			T=task.T, 
			eps=eps, 
			hmc=True
		)

	    hmc_x = tf.placeholder(tf.float32, shape=(None, x_dim))
	    _, _, _, hmc_MH = propose(hmc_x, hmc_dynamics, do_mh_step=True)
	    
	    HMC_samples = []
	    samples = np.copy(init_samples)
	    for t in range(args.eval_steps):
	        HMC_samples.append(np.copy(samples))
	        samples = sess.run(hmc_MH[0], {hmc_x: samples})

	    all_hmc.append(np.array(HMC_samples))

	for i, H in enumerate(all_hmc):
		plt.plot(
			np.abs([autocovariance((H-mu) / std, tau=t) for t in range(args.eval_steps - 1)]),
			label='$\epsilon=%g$' % eps_eval[i]
		)

	plt.plot(
		np.abs([autocovariance((F-mu) / std, tau=t) for t in range(args.eval_steps - 1)]),
		label='L2HMC',
	)
	plt.xlabel('# MH steps')
	plt.ylabel('Autocovariance')
	plt.legend()

	plt.savefig('%s/autocorrelation.png' % logdir)

	ess = {
		'L2HMC': effective_n(F),
		'HMC': {eps_eval[i]: effective_n(all_hmc[i]) for i in range(len(eps_eval))}
	}

	with open('%s/ess.txt' % logdir, 'w') as f:
		json.dump(ess, f)







