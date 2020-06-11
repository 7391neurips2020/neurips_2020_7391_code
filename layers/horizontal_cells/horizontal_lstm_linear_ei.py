import collections
import math
import numpy as np

import tensorflow as tf
from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables  # pylint: disable=unused-import
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

class Horizontal_Linear_EI():
  # Class to define a custom
  # LSTM cell for accessing
  # intermediate states of
  # the LSTM

  def __init__(self,
                input_shape,
                output_channels,
                kernel_shape,
                inh_mult,
                exc_mult,
                use_bias=True,
                initializers=None,
                activation=None,
                batch_size=None,
                timesteps=None,
                activation_hor=True,
                skip_connection=False,
                name="horizontal_cell"):
    self._input_shape = input_shape
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._inh_mult = inh_mult
    self._exc_mult = exc_mult
    self._use_bias = use_bias
    self._skip_connection = skip_connection
    self.activation = activation
    self.initializers = initializers
    self.timesteps = timesteps
    self.activation_hor = activation_hor
    state_size = tensor_shape.TensorShape(
        self._input_shape[:-1] + [self._output_channels])
    self._state_size = rnn_cell_impl.LSTMStateTuple(state_size,
                                                    state_size)
    self._output_size = tensor_shape.TensorShape(
                                                self._input_shape[:-1]
                                                + [self._output_channels])

  @property
  def output_size(self):
      return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def call(self, inputs, state, scope=None):
    """`cell` contains the contextually
    modulated V1 responseself.
    modulation occurs through the following
    rule:
    Xt + ((ht*He)-(ht*Hi))/((1-ht*He)x(ht*Hi)"""

    activation_fn = get_activation(self.activation)
    itr = tf.constant(0,shape=())

    l_outputs = tf.TensorArray(dtype=tf.float32,
                            size=1, dynamic_size=True,
                            clear_after_read=False)

    l_H_exc = tf.TensorArray(dtype=tf.float32,
                            size=1, dynamic_size=True,
                            clear_after_read=False)
    l_H_inh = tf.TensorArray(dtype=tf.float32,
                            size=1, dynamic_size=True,
                            clear_after_read=False)

    def body(itr,
            inputs,
            l_outputs,
            l_H_exc,
            l_H_inh,
            state):
      """Main computation of horizontal cell
      takes place within this body()
      :param itr: Iteration count
      :param inputs: Minibatch of inputs
                    of shape [t,n,h,w,c]
      :param state: Current state of the
                    RNN in LSTM style
      """
      cell, hidden = state
      input = inputs[itr,:,:,:,:]
      new_hidden = _conv([input, hidden],
                         self._kernel_shape,
                         self._output_channels,
                         self._use_bias,
                         inh_mult=self._inh_mult,
                         exc_mult=self._exc_mult,
                         activation=self.activation,
                         initializers=self.initializers,
                         )
      conv_x, conv_h, conv_exc, conv_inh = new_hidden
      X_i, X_f, X_c, X_o = array_ops.split(conv_x,
                                        num_or_size_splits=4,
                                        axis=-1,
                                        )
      H_i, H_f, H_o = array_ops.split(conv_h,
                                    num_or_size_splits=3,
                                    axis=-1,
                                    )
      H_exc, H_inh = conv_exc, conv_inh
      if self.activation_hor:
          print 'Added sigmoid to horizontal connections'
          H_exc = math_ops.sigmoid(H_exc)
          H_inh = math_ops.sigmoid(H_inh)
      # computing the gates
      input_gate = tf.math.add(X_i,H_i,name='input_gate')
      forget_gate = tf.math.add(X_f,H_f,name='forget_gate')
      output_gate = tf.math.add(X_o,H_o,name='output_gate')

      # computing horizontal push/pull
      # Directly operate on excitatory and inhibitory
      # states instead of operating on H(t-1)
      l_push_pull = X_c + H_exc - H_inh
      # nl_push_pull = (1.-H_exc)*H_inh
      nl_push_pull = None
      if nl_push_pull is not None:
        context_mod = tf.math.divide(
                                      l_push_pull,
                                      nl_push_pull,
                                      name='contextual_modulation'
                                      )
      else:
        context_mod = tf.identity(l_push_pull)
      new_input = X_c + context_mod
      new_cell = math_ops.sigmoid(forget_gate) * cell
      new_cell += math_ops.sigmoid(input_gate) * math_ops.tanh(new_input)
      if activation_fn:
        output = activation_fn(new_cell) * math_ops.sigmoid(output_gate)
      else:
        output = new_cell * math_ops.sigmoid(output_gate)
      if self._skip_connection:
        output = array_ops.concat([output, input], axis=-1)
      l_outputs = l_outputs.write(itr, output)
      l_H_exc = l_H_exc.write(itr, H_exc)
      l_H_inh = l_H_inh.write(itr, H_inh)
      new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)
      # Incrementing loop counter
      itr = itr + 1
      return [itr, inputs,
            l_outputs, l_H_exc,
            l_H_inh, new_state]

    def cond(itr, inputs, l_outputs,
            l_H_exc, l_H_inh,
            state):
      return tf.less(itr, self.timesteps)

    (itr, inputs, L_outputs,
    L_H_exc, L_H_inh, new_state) = tf.while_loop(
                                            cond,
                                            body,
                                            [itr, inputs, l_outputs,
                                            l_H_exc, l_H_inh,
                                            state],
                                            )
    L_outputs = L_outputs.stack(name='Context')
    L_H_exc = L_H_exc.stack(name='Hidden_Excitation')
    L_H_inh = L_H_inh.stack(name='Hidden_Inhibition')
    print 'Added activation: %s'%(self.activation),activation_fn
    to_return = (inputs, L_outputs,
                 L_H_exc, L_H_inh,
                 new_state)
    return to_return


def get_activation(activation):
  """Function that returns an instance
  of an activation function"""
  if activation == 'relu':
      return tf.nn.relu
  if activation == 'sigmoid':
      return tf.nn.sigmoid
  if activation == 'tanh':
      return tf.nn.tanh


def _conv(args, filter_size, output_channels, bias,
            inh_mult=1.5, exc_mult=3, bias_start=0.0,
            activation=None, initializers=None,
            dtype=tf.float32):
  """Convolution.
  Args:
    args: a Tensor or a list of Tensors of dimension 3D, 4D or 5D,
    batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    output_channels: int, number of convolutional kernels.
    bias: Whether to use biases in the convolution layer.
    bias_start: starting value to initialize the bias; 0 by default.
  Returns:
    A 3D, 4D, or 5D Tensor with shape [batch ... num_features]
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  def get_initializer():
    initializer = tf.glorot_uniform_initializer(
                          seed=None,
                          dtype=tf.float32,
                          )
    return initializer

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
  shape_length = len(shapes[0])
  n_args = len(args)
  input, hidden = args
  if n_args > 2:
    raise ValueError("Expected only two "
      "arguments (input, hidden)")

  for shape in shapes:
    if len(shape)!=4:
      raise ValueError("Expected only 4-D arrays of "
        "form [n,h,w,c] for performing 2D convolutions"
        )
    if len(shape) != len(shapes[0]):
      raise ValueError("Conv Linear expects all args "
                       "to be of same Dimension: %s" % str(shapes))
  # import ipdb; ipdb.set_trace()
  x_arg_depth = shapes[0][-1]
  h_arg_depth = shapes[1][-1]
  conv_op = nn_ops.conv2d
  strides = shape_length * [1]
  # TODO: Check extent of long-range inhibition
  f_h, f_w = filter_size
  f_h_inh, f_w_inh = int(f_h*inh_mult), int(f_w*inh_mult)
  f_h_exc, f_w_exc = int(f_h*exc_mult), int(f_w*exc_mult)
  filter_size_inh = [f_h_inh, f_w_inh]
  filter_size_exc = [f_h_exc, f_w_exc]

  # Build input and hidden kernels
  x_kernel = vs.get_variable(
      "input_kernel", filter_size + [x_arg_depth, output_channels*4],
      initializer=get_initializer(),
      dtype=tf.float32)

  # Build hidden state kernels
  h_kernel_gates = vs.get_variable(
      "hidden_kernel_g", filter_size + [x_arg_depth, output_channels*3],
      initializer=get_initializer(),
      dtype=tf.float32)
  # TODO: find optimal l1 strength
  h_kernel_inh = vs.get_variable(
      "hidden_kernel_inh", filter_size_inh + [x_arg_depth, output_channels],
      initializer=get_initializer(),
      regularizer=tf.contrib.layers.l1_regularizer(1e-2),
      dtype=tf.float32)
  h_kernel_exc = vs.get_variable(
      "hidden_kernel_exc", filter_size_exc + [x_arg_depth, output_channels],
      initializer=get_initializer(),
      regularizer=tf.contrib.layers.l1_regularizer(1e-2),
      dtype=tf.float32)

  res_x = conv_op(input,
          x_kernel,
          strides,
          padding="SAME")

  res_h_gates = conv_op(hidden,
                h_kernel_gates,
                strides,
                padding="SAME")
  res_h_inh = conv_op(hidden,
              h_kernel_inh,
              strides,
              padding="SAME")
  res_h_exc = conv_op(hidden,
              h_kernel_exc,
              strides,
              padding="SAME")
  if not bias:
    return res
  bias_input = vs.get_variable(
      "biases_input", [output_channels*4],
      dtype=tf.float32,
      initializer=init_ops.constant_initializer(
                        bias_start,
                        dtype=dtype))
  bias_hidden_gates = vs.get_variable(
        "biases_hidden_g", [output_channels*3],
        dtype=tf.float32,
        initializer=init_ops.constant_initializer(
                          bias_start,
                          dtype=dtype))
  bias_hidden_exc = vs.get_variable(
        "biases_hidden_e", [output_channels],
        dtype=tf.float32,
        initializer=init_ops.constant_initializer(
                          bias_start,
                          dtype=dtype))
  bias_hidden_inh = vs.get_variable(
        "biases_hidden_i", [output_channels],
        dtype=tf.float32,
        initializer=init_ops.constant_initializer(
                          bias_start,
                          dtype=dtype))
  res_input = tf.math.add(res_x,
                          bias_input,
                          name='conv_input_gates'
                          )
  res_hidden_gates = tf.math.add(res_h_gates,
                                bias_hidden_gates,
                                name='conv_hidden_gates'
                                )
  res_hidden_inh = tf.math.add(res_h_inh,
                                bias_hidden_inh,
                                name='conv_hidden_inh'
                                )
  res_hidden_exc = tf.math.add(res_h_exc,
                                bias_hidden_exc,
                                name='conv_hidden_exc'
                                )
  return (res_input, res_hidden_gates,
            res_hidden_exc, res_hidden_inh)

def main():
  bs, ts = 10, 8
  inp = tf.placeholder(tf.float32, shape=[bs,100,100,32])
  batch_size = inp.shape[0]
  input_tile = tf.tile(inp, [ts,1,1,1])
  input = tf.reshape(input_tile, [ts,bs,100,100,32],
                        name='Input_Tensor')
  state_c = tf.placeholder(tf.float32,
                            shape=[bs,100,100,32],
                            name='cell')
  state_h = tf.placeholder(tf.float32,
                            shape=[bs,100,100,32],
                            name='hidden')
  state = tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h)
  hcell = Horizontal_Custom(input_shape=[100,100,32],
                          output_channels=32,
                          kernel_shape=[3,3],
                          inh_mult=1.5,
                          exc_mult=3,
                          timesteps=ts,
                          activation='tanh',
                          )
  all_returned = hcell.call(inputs=input, state=state)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  feed_dict = {inp:np.random.random(inp.shape),
                state_c:np.random.random(state_c.shape),
                state_h:np.random.random(state_h.shape),
                }
  returned = sess.run(all_returned, feed_dict=feed_dict)
  import ipdb; ipdb.set_trace()

if __name__=='__main__':
  main()
