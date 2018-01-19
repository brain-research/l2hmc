import tensorflow as tf
import numpy as np
from utils.dynamics import Dynamics
from utils.sampler import propose
import matplotlib.pyplot as plt

def plot_grid(S, width=8):
    sheet_width = width
    plt.figure(figsize=(12, 12))
    for i in xrange(S.shape[0]):
        plt.subplot(sheet_width, sheet_width, i + 1)
        plt.imshow(S[i], cmap='gray')
        plt.grid('off')
        plt.axis('off')
        
def plot_line(S):
    sheet_width = S.shape[0]
    plt.figure(figsize=(16, 3))
    for i in xrange(S.shape[0]):
        plt.subplot(1, sheet_width, i + 1)
        plt.imshow(S[i], cmap='gray')
        plt.grid('off')
        plt.axis('off')
        
def get_hmc_samples(x_dim, eps, energy_function, sess, T=10, steps=200, samples=None):
    hmc_dynamics = Dynamics(x_dim, energy_function, T=T, eps=eps, hmc=True)
    hmc_x = tf.placeholder(tf.float32, shape=(None, x_dim))
    Lx, _, px, hmc_MH = propose(hmc_x, hmc_dynamics, do_mh_step=True)
    
    if samples is None:
        samples = gaussian.get_samples(n=200)
        
    final_samples = []
    
    for t in range(steps):
        final_samples.append(np.copy(samples))
        Lx_, px_, samples = sess.run([Lx, px, hmc_MH[0]], {hmc_x: samples})
        
    return np.array(final_samples)
