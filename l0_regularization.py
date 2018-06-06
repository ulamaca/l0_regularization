import tensorflow as tf
import numpy as np



def tf_hard_sigmoid(u):
    h = tf.minimum(1.0, tf.maximum(0.0, u))
    return h

def tf_stretch(s, interval):
    """
    s: 1-d a tensor
    interval: [a,b], the lower/upper bound of the interval to stretch up
    """
    s_bar = s * (interval[1] - interval[0]) + interval[0]
    return s_bar

def tf_concrete_transoform(u, alpha, beta):
    '''
    :param u: a random variable, usually from uniform distribution
    :param alpha:
    :param beta:
    :return:
    '''
    ct = tf.nn.sigmoid((tf.log(u) - tf.log(1 - u) + tf.log(alpha)) / beta)
    return ct

def L0_loss(latent_batch, latent_dim, reg_const):
    '''
    compute the gating variable and regularization terms
    :return: z_gate: the gating variables; reg: the sum of all regularization terms; alpha: the sparse parameters to tune
    '''
    interval = [-0.1, 1.1]
    u = tf.random_uniform(shape=[latent_dim], dtype=tf.float32)
    # c = log(\alpha)
    c = tf.get_variable(tf.random_normal('gating_node', shape=[latent_dim], mean=np.log(3 / 7), stddev=1e-3))  # the parameter to tune
    alpha = tf.exp(c)
    beta = 2/3
    s = tf_concrete_transoform(u, alpha, beta)

    s_bar = tf_stretch(s, interval)
    s_bar_pred = tf_stretch(tf.nn.sigmoid(c), interval)

    latent_gated_training = tf_hard_sigmoid(s_bar) * latent_batch
    latent_gated_prediction = tf_hard_sigmoid(s_bar_pred) * latent_batch
    latent_batch = (latent_gated_training, latent_gated_prediction)
    add_losses = tf.nn.sigmoid(c - beta * (tf.log(-interval[0]) - tf.log(interval[1])))
    l0_loss = tf.reduce_sum(add_losses)
    tf.summary.scalar('l0_loss', l0_loss)
    add_loss = reg_const * l0_loss

    train_op_checkers = tf.no_op() # useless but for spec requirement

    return add_loss, train_op_checkers, latent_batch

