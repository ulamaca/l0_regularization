import tensorflow as tf
import numpy as np
from tensorflow.python.platform import tf_logging as logging
import numbers
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops

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

# todo: to be deprecated
def l0_regularization(latent_batch, latent_dim, reg_const):
    '''
    compute the gating variable and regularization terms
    :return: z_gate: the gating variables; reg: the sum of all regularization terms; alpha: the sparse parameters to tune
    '''
    interval = [-0.1, 1.1]
    u = tf.random_uniform(shape=[latent_dim], dtype=tf.float32)
    # c = log(\alpha)
    # c = tf.get_variable('gating_node', tf.random_normal(shape=[latent_dim], mean=np.log(3 / 7), stddev=1e-3))  # the parameter to tune
    c = tf.Variable(tf.random_normal(shape=[latent_dim], mean=np.log(3 / 7), stddev=1e-3), name='gating_node', dtype=tf.float32)  # todo, use tf.get_variable the parameter to tune
    alpha = tf.exp(c)
    beta = 2/3
    s = tf_concrete_transoform(u, alpha, beta)

    s_bar = tf_stretch(s, interval)
    s_bar_pred = tf_stretch(tf.nn.sigmoid(c), interval)

    latent_gated_training = tf_hard_sigmoid(s_bar) * latent_batch
    latent_gated_prediction = tf_hard_sigmoid(s_bar_pred) * latent_batch
    #is_training = tf.placeholder(tf.bool, shape=0)
    def latent_batch(is_training):
        return tf.cond(is_training, lambda: latent_gated_training, lambda: latent_gated_prediction)
    add_losses = tf.nn.sigmoid(c - beta * (tf.log(-interval[0]) - tf.log(interval[1])))
    l0_loss = tf.reduce_sum(add_losses)
    tf.summary.scalar('l0_loss', l0_loss)
    add_loss = reg_const * l0_loss

    return add_loss, latent_batch

def l0_computation(tensor,
                   interval=[-0.1, 1.1],
                   mu_c=np.log(3 / 7),
                   sigma_c=1e-3,
                   beta=2/3):
    tensor_shape = tensor.get_shape()
    u = tf.random_uniform(shape=tensor_shape, dtype=tf.float32)

    c = tf.Variable(tf.random_normal(shape=tensor_shape, mean=mu_c, stddev=sigma_c), 
                    name='mask',
                    collections=['l0_vars'],
                    dtype=tf.float32)  # todo, use tf.get_variable the parameter to tune
    '''
    c = tf.get_variable(name='mask',
                        shape=tensor_shape,
                        initializer=tf.random_normal_initializer(mean=mu_c, stddev=sigma_c),
                        collections=['l0_vars'],
                        dtype=tf.float32 # todo: might be optional
                        )
    '''
    alpha = tf.exp(c)
    s = tf_concrete_transoform(u, alpha, beta)

    s_bar = tf_stretch(s, interval)
    s_bar_pred = tf_stretch(tf.nn.sigmoid(c), interval)

    l0_tensor_training = tf_hard_sigmoid(s_bar) * tensor
    l0_tensor_prediction = tf_hard_sigmoid(s_bar_pred) * tensor

    # is_training = tf.placeholder(tf.bool, shape=0)
    '''
    def l0_maksed_tensor(is_training):
        return tf.cond(is_training,
                       lambda: l0_tensor_training, lambda: l0_tensor_prediction,
                       name="l0_masked_XXX") #todo: add the name of the tensor's original name
    '''
    l0_maksed_tensor = None

    add_losses = tf.nn.sigmoid(c - beta * (tf.log(-interval[0]) - tf.log(interval[1])))
    l0_loss = tf.reduce_sum(add_losses)
    add_loss = l0_loss

    return add_loss, l0_maksed_tensor

def l0_regularizer(scale, scope=None):
    """Returns a function that can be used to apply L2 regularization to weights.
      Small values of L2 can help prevent overfitting the training data.
      Args:
        scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
        scope: An optional scope name.
      Returns:
        A function with signature `l2(weights)` that applies L2 regularization.
      Raises:
        ValueError: If scale is negative or if scale is not a float.
      """
    if isinstance(scale, numbers.Integral):
        raise ValueError('scale cannot be an integer: %s' % (scale,))
    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                             scale)
        if scale == 0.:
            logging.info('Scale of 0 disables regularizer.')
            return lambda _: None

    def l0(weights):
        """Applies l2 regularization to weights."""
        with ops.name_scope(scope, 'l0_regularizer', [weights]) as name:
            print(name)
            my_scale = ops.convert_to_tensor(scale,
                                             dtype=weights.dtype.base_dtype,
                                             name='scale')
            # l0_loss, _ = l0_computation(weights)
            l0_loss, debug = l0_computation(weights)
            return standard_ops.multiply(my_scale, l0_loss, name=name)

    return l0
