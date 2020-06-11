import collections
import math
import numpy as np


import tensorflow as tf
# from tensorflow.contrib.compiler import jit
# from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import initializers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging


conv_op = nn_ops.conv2d

def get_initializer(dtype=tf.float32):
  initializer = tf.compat.v1.variance_scaling_initializer(
                        seed=None,
                        dtype=dtype,
                        )
  return initializer


def get_activation(activation):
  """Function that returns an instance
  of an activation function"""
  if activation == 'relu':
      return tf.nn.relu
  if activation == 'sigmoid':
      return tf.nn.sigmoid
  if activation == 'tanh':
      return tf.nn.tanh


class ConvGRUCell(tf.nn.rnn_cell.RNNCell):
  """A GRU cell with convolutions instead of multiplications."""

  def __init__(self, shape, filters, kernel, activation='relu', normalize=True, data_format='channels_last', reuse=None):
    super(ConvGRUCell, self).__init__(_reuse=reuse)
    self._filters = filters
    self._kernel = kernel
    self._activation = get_activation(activation)
    self._normalize = normalize
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def call(self, x, h):
    channels = x.shape[self._feature_axis].value
    dtype = x.dtype
    with tf.variable_scope('gates'):
      inputs = tf.concat([x, h], axis=self._feature_axis)
      n = channels + self._filters
      m = 2 * self._filters if self._filters > 1 else 2
      W = vs.get_variable('kernel', self._kernel + [n, m],
			initializer=get_initializer(dtype),
			dtype=dtype)
      y = conv_op(inputs, W, padding='SAME', data_format=self._data_format)
      y += vs.get_variable('bias', [m], initializer=init_ops.constant_initializer(
                                                    0.0, dtype=dtype))
      r, u = tf.split(y, 2, axis=self._feature_axis)
      r, u = math_ops.sigmoid(r), math_ops.sigmoid(u)

    with tf.variable_scope('candidate'):
      inputs = tf.concat([x, r * h], axis=self._feature_axis)
      n = channels + self._filters
      m = self._filters
      W = vs.get_variable('kernel', self._kernel + [n, m],
                            initializer=get_initializer(dtype),
                            dtype=dtype)
      y = conv_op(inputs, W, padding='SAME', data_format=self._data_format)
      y += vs.get_variable('bias', [m], initializer=tf.zeros_initializer())
      h = u * h + (1 - u) * self._activation(y)

    return h, h




