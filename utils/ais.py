import tensorflow as tf
import numpy as np

from utils import tf_accept

def ais_estimate(
        init_energy, 
        final_energy, 
        anneal_steps, 
        initial_x, 
        step_size=0.5, 
        leapfrogs=25, 
        x_dim=50
    ):
    beta = tf.linspace(0., 1., anneal_steps+1)[1:]
    beta_diff = beta[1] - beta[0]

    def body(a, beta):
        curr_energy = lambda z: (1-beta) * init_energy(z) + (beta) * final_energy(z)
        last_x = a[1]
        w = a[2]
        w = w + beta_diff * (- final_energy(last_x) + init_energy(last_x))
        dynamics = Dynamics(x_dim, energy_function=curr_energy, eps=step_size, hmc=True, T=leapfrogs)
        Lx, _, px = dynamics.forward(last_x)
        updated_x = tf_accept(last_x, Lx, px)
        return (px, updated_x, w)

    alpha, x, w = tf.scan(body, beta, (tf.zeros_like(initial_x[:, 0]),
                        initial_x, tf.zeros_like(initial_x[:, 0])))
    
    return tf.reduce_logsumexp(w[-1]) - tf.log(tf.shape(initial_x)[0])