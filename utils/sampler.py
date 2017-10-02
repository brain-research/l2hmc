import tensorflow as tf
import numpy as np

def propose(x, dynamics, init_v=None, aux=None, do_mh_step=False, log_jac=False):
	if dynamics.hmc:
		Lx, Lv, px = dynamics.forward(x, init_v=init_v, aux=aux)
		return Lx, Lv, px, [tf_accept(x, Lx, px)]
	else:
		# sample mask for forward/backward
		mask = tf.cast(tf.random_uniform((tf.shape(x)[0], 1), maxval=2, dtype=tf.int32), tf.float32)
		Lx1, Lv1, px1 = dynamics.forward(x, aux=aux, log_jac=log_jac)
		Lx2, Lv2, px2 = dynamics.backward(x, aux=aux, log_jac=log_jac)

        Lx = mask * Lx1 + (1 - mask) * Lx2

        Lv = None
        if init_v is not None:
        	Lv = mask * Lv1 + (1 - mask) * Lv2

        px = tf.squeeze(mask, axis=1) * px1 + tf.squeeze(1 - mask, axis=1) * px2

        outputs = []

        if do_mh_step:
        	outputs.append(tf_accept(x, Lx, px))

        return Lx, Lv, px, outputs

def tf_accept(x, Lx, px):
    mask = (px - tf.random_uniform(tf.shape(px)) >= 0.)
    return tf.where(mask, Lx, x)