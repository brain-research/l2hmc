"""TODO(danilevy): DO NOT SUBMIT without one-line documentation for gen_sh.

TODO(danilevy): DO NOT SUBMIT without a detailed description of gen_sh.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google3.pyglib import app
from google3.pyglib import flags

FLAGS = flags.FLAGS

import numpy as np

TASKS = ['gaussian_1', 'gaussian_2', 'mog']
LOSSES = ['inv', 'logsumexp']
OPT = ['adam', 'rmsprop']

DEFAULT_HPARAMS = tf.contrib.training.HParams(
    learning_rate=0.001,
    hidden_sizes=[10, 10],
    optimizer='adam',
    loss='inv',
    training_steps=10000,
    eval_steps=2000,
    batch_size=128,
    task='gaussian_1',
)
def main(argv):
  for _ in range(5):
    lr = 10 ** np.random.uniform(low=-5, high=-2, size=())
    opt = np.random.choice(OPT)
    loss = np.random.choice(LOSSES)
    task = np.random.choice(TASKS)
    exp_id = 0

    string = ("/google/data/ro/teams/traino/borgcfg --skip_confirmation"
    "--vars \"exp_id=%d,cell=ok,gpu_priority=115,use_allocator=true,hparams='learning_rate=%.4e,optimizer=%s,loss=%s,task=%s'\""
    "experimental/users/danilevy/l2hmc/launch.borg reload") % (exp_id, lr, opt, loss, task)

if __name__ == '__main__':
  app.run(main)
