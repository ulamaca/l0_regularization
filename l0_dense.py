from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.python.layers.core import Dense # todo-debug1: is this import correct?
from tensorflow.python.ops import init_ops
from tensorflow.python.layers import base
from tensorflow.python.framework import tensor_shape
from l0_regularization import l0_regularizer


###########################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tb_graph_jupyter import show_graph
# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
from l0_regularization import l0_regularizer

# Parameter
learning_rate = 0.01
training_epochs = 10
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)
reg_const = 1e-6 # ~ 1e3/n == 1e5-1e6 for MNIST
regularizer = l0_regularizer(reg_const)



###############################33

# todo 2: create a l0_Dense class inherited from Dense
class L0Dense(Dense):
  def __init__(self, units,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(L0Dense, self).__init__(units,
                                  activation=activity_regularizer,
                                  use_bias=use_bias,
                                  kernel_initializer=kernel_initializer,
                                  bias_initializer=bias_initializer,
                                  kernel_regularizer=kernel_regularizer,
                                  bias_regularizer=bias_regularizer,
                                  activity_regularizer=activity_regularizer,
                                  kernel_constraint=kernel_constraint,
                                  bias_constraint=bias_constraint,
                                  trainable=trainable,
                                  name=name,
                                  **kwargs)
    self.units = units
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint
    self.input_spec = base.InputSpec(min_ndim=2)

  def call(self, inputs):
      l0_var_list=tf.get_collection('l0_vars', scope=self.scope_name)
      print(l0_var_list)

      # todo to imporve:
        # 1 self.kernel_regularizer == l0 # ideal format, since it is possible to combine l0 with other regularization
        # 2 the way I take the masekd variables out
      # create masked kernel/bias
      kernl_0 = self.kernel
      if self.kernel_regularizer is not None:
          print([var for var in l0_var_list if 'kernel' in var.name])
          kernel_mask = [var for var in l0_var_list if 'kernel' in var.name][0]
          masked_kernel = tf.identity(tf.multiply(self.kernel, kernel_mask), name='l0_masked_kernel')
          self.kernel = masked_kernel


      bias_0 = self.bias
      if self.bias_regularizer is not None:
          bias_mask = [var for var in l0_var_list if 'bias' in var.name][0]
          masked_bias = tf.identity(tf.multiply(self.bias, bias_mask), name='l0_masked_bias')
          self.bias = masked_bias

      output = super(L0Dense, self).call(inputs)
      # todo Maybe I can also set the l0_layer here
      #if self.activity_regularizer is not None:

      # change back to the orignal one
      self.kernel = kernl_0
      self.bias = bias_0

      return output

# todo: I only change one line of code from the original dense function, could i do it in a esaier way?
    # **kwargs # use this to first simplify the code
def l0_dense(
      inputs, units,
      activation=None,
      use_bias=True,
      kernel_initializer=None,
      bias_initializer=init_ops.zeros_initializer(),
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      trainable=True,
      # is_training=True,
      name=None,
      reuse=None):
  """Functional interface for the densely-connected layer.

  This layer implements the operation:
  `outputs = activation(inputs.kernel + bias)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `kernel` is a weights matrix created by the layer,
  and `bias` is a bias vector created by the layer
  (only if `use_bias` is `True`).

  Arguments:
    inputs: Tensor input.
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the default
      initializer used by `tf.get_variable`.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor the same shape as `inputs` except the last dimension is of
    size `units`.

  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = L0Dense(units,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                trainable=trainable,
                name=name,
                dtype=inputs.dtype.base_dtype,
                _scope=name,
                _reuse=reuse)
  return layer.apply(inputs)

def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = l0_dense(x, n_hidden_1, kernel_regularizer=regularizer)
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = l0_dense(layer_1, n_hidden_2, kernel_regularizer=regularizer)
    return layer_2

# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = l0_dense(x, n_hidden_1, kernel_regularizer=regularizer)
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = l0_dense(layer_1, n_input, kernel_regularizer=regularizer)
    return layer_2


if __name__ == "__main__":

    # the 1st test
    '''
    n_input = 784
    n_hidden_1 = 256
    c_l0 = 1e-3
    x = tf.placeholder("float", [None, n_input])
    tmp = L0Dense(n_hidden_1,
                  bias_regularizer=l0_regularizer(c_l0),
                  kernel_regularizer=l0_regularizer(c_l0))

    x1 = tmp.apply(x)
    '''


    # the 2nd test
    tf.reset_default_graph()
    X = tf.placeholder("float", [None, n_input])
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    # Construct model
    with tf.variable_scope('encoder'):
        encoder_op = encoder(X)

    with tf.variable_scope('decoder'):
        decoder_op = decoder(encoder_op)

    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X

    # Define loss and optimizer, minimize the squared error
    losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) + tf.reduce_sum(losses)
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    # collect variables to observe:

    # Initializing the variables
    init = tf.global_variables_initializer()
    print(tf.trainable_variables())
    # Launch the graph
    # Using InteractiveSession (more convenient while using Notebooks)
    sess = tf.InteractiveSession()
    sess.run(init)

    total_batch = int(mnist.train.num_examples / batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, is_training: True})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(c))
    # W_to_check.eval()
    print("Optimization Finished!")

