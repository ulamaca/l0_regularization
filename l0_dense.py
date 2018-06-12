from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.python.layers.core import Dense # todo-debug1: is this import correct?
from tensorflow.python.ops import init_ops
from tensorflow.python.layers import base
from tensorflow.python.framework import tensor_shape
from .l0_regularization import l0_regularizer



def get_tensor_by_name(name):
    return tf.get_default_graph().get_tensor_by_name(name)

# todo 2: create a l0_Dense class inherited from Dense
class L0Dense(Dense):
  def __init__(self, units, is_training,
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
    self.is_training = is_training
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint
    self.input_spec = base.InputSpec(min_ndim=2)

  def call(self, inputs):
      # todo to imporve:
        # 1 self.kernel_regularizer == l0 # ideal format, since it is possible to combine l0 with other regularization
        # 2 the way I take the masekd variables out

      l0_relative_path = '/Regularizer/l0_regularizer/'
      l0_absolute_path = self.scope_name
      # create masked kernel/bias
      kernel_0 = self.kernel
      if self.kernel_regularizer is not None:
          trng_masked_kernel = get_tensor_by_name(l0_absolute_path + "/kernel" + l0_relative_path + "trng_mask:0")
          pred_masked_kernel = get_tensor_by_name(l0_absolute_path + "/kernel" + l0_relative_path + "pred_mask:0")
          masked_kernel = tf.cond(self.is_training,
                                  lambda: trng_masked_kernel,
                                  lambda: pred_masked_kernel, name='l0_masked_kernel')

          # Alternative Masking: better learning result but not theoretically correct
          # trng_masked_kernel = get_tensor_by_name(l0_absolute_path + "/kernel" + l0_relative_path + "el0n_mask:0")
          # pred_masked_kernel = get_tensor_by_name(l0_absolute_path + "/kernel" + l0_relative_path + "el0n_mask:0")
          # masked_kernel = tf.cond(self.is_training,
          #                         lambda: trng_masked_kernel * kernel_0,
          #                         lambda: pred_masked_kernel * kernel_0, name='l0_masked_kernel')


          self.kernel = masked_kernel

      bias_0 = self.bias
      if self.bias_regularizer is not None:
          trng_masked_bias = get_tensor_by_name(l0_absolute_path + "/bias" + l0_relative_path + "trng_mask:0")
          pred_masked_bias = get_tensor_by_name(l0_absolute_path + "/bias" + l0_relative_path + "pred_mask:0")
          masked_bias = tf.cond(self.is_training,
                                lambda: trng_masked_bias,
                                lambda: pred_masked_bias, name='l0_masked_bias')

          self.bias = masked_bias

      output = super(L0Dense, self).call(inputs)

      # todo Maybe I can also set the l0_layer here, if self.activity_regularizer is not None:
      # change back to the orignal one
      self.kernel = kernel_0
      self.bias = bias_0

      return output

# todo: I only change one line of code from the original dense function, could i do it in a esaier way?
    # **kwargs # use this to first simplify the code

def l0_dense(
      inputs, units, is_training,
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
  layer = L0Dense(units, is_training,
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


if __name__ == "__main__":

    # the 1st test
    n_input = 784
    n_hidden_1 = 256
    c_l0 = 1e-3
    x = tf.placeholder("float", [None, n_input])
    tmp = L0Dense(n_hidden_1,
                  bias_regularizer=l0_regularizer(c_l0),
                  kernel_regularizer=l0_regularizer(c_l0))

    x1 = tmp.apply(x)

