import numpy as np
import tensorflow as tf
from utils.tf_utils import  *
from layers.horizontal_cells.v1net_cell import V1Net_cell
from layers.horizontal_cells.v1net_bn_cell import V1Net_BN_cell
from layers.horizontal_cells.hgru_cell import ConvGRUCell as grucell

def lstm_layer(X, ts, n_filt, k_hw, filter_reg=None, name=None):
    # Recurrent ConvLSTM layer for retina
    n,h,w,c = X.shape.as_list()
    lstm_tile = tf.tile(X, [ts,1,1,1])
    lstm_input = tf.reshape(lstm_tile, [-1,ts,h,w,c])
    lstm_v1 = tf.contrib.rnn.Conv2DLSTMCell(
                                input_shape=[h,w,c],
                                output_channels=n_filt,
                                kernel_shape=[k_hw, k_hw]
                                )
    _, (X_c, X_h) = tf.nn.dynamic_rnn(cell=lstm_v1,
                                inputs=lstm_input,
                                sequence_length=[ts]*n,
                                dtype=tf.float32,
                                )
    return X_h


def hgru_layer(X, ts, n_filt, k_hw, filter_reg=None, name=None):
    n, h, w, c = X.shape.as_list()
    t, bs = ts, n
    ones = tf.ones([t,1,1,1,1],dtype=tf.float32)
    hgru_input = ones * X
    hgru_input = tf.transpose(hgru_input, [1,0,2,3,4])
    cell = grucell(shape=[h,w], filters=n_filt,
                    kernel=[k_hw,k_hw]
                    )
    outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                inputs=hgru_input,
                                dtype=tf.float32)
    return state 


def v1net_shnt_layer(X, ts, n_filt, k_hw, filter_init, filter_reg=None,
                     inh_mult=1.5, exc_mult=3, pointwise=False, v1_act='relu', 
                     name=None):
    ## Horizontal inhibitory and
    ## excitatory connections
    ## influencing classical RF
    ## through V1net. Implementing
    ## correlation norm to enable
    ## like-tuned inhibition/excitation
    n,h,w,c = X.shape.as_list()
    print(n,h,w,c)
    t, bs = ts, n
    # v1net_tile = tf.tile(X, [t,1,1,1])
    # v1net_input = tf.reshape(v1net_tile, [n,t,h,w,c])
    ones = tf.ones([t,1,1,1,1],dtype=tf.float32)
    v1net_input = ones*X
    v1net_input = tf.transpose(v1net_input, [1,0,2,3,4])
    state_c = tf.zeros([n,h,w,n_filt], dtype=X.dtype, name='init_c')
    state_h = tf.zeros([n,h,w,n_filt], dtype=X.dtype, name='init_h')
    #state_c = tf.zeros_like(X, name='init_c', dtype=tf.float32)
    #state_h = tf.zeros_like(X, name='init_h', dtype=tf.float32)
    state = tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h)
    v1net_cell = V1Net_cell(
                            input_shape=[h,w,c],
                            output_channels=n_filt,
                            kernel_shape=[k_hw,k_hw],
                            inh_mult=inh_mult,
                            exc_mult=exc_mult,
                            activation=v1_act,
                            timesteps=t,
                            activation_hor=True,
                            pointwise=pointwise,
                            )
    v1net_out = tf.nn.dynamic_rnn(cell=v1net_cell,
                                inputs=v1net_input,
                                initial_state=state,
                                dtype=tf.float32,
                                )
    new_state_c, new_state_h = v1net_out[1]
    return new_state_h

def v1net_shunt_bn_layer(X, ts, n_filt, k_hw, training, filter_init, filter_reg=None,
                    inh_mult=1.5, exc_mult=3, pointwise=False, v1_act='relu', 
                    name=None):
    ## Horizontal inhibitory and
    ## excitatory connections
    ## influencing classical RF
    ## through V1net. Implementing
    ## correlation norm to enable
    ## like-tuned inhibition/excitation
    print('Inside neurips_2020_7391_code w/ batchnormalization')
    n,h,w,c = X.shape.as_list()
    print(n,h,w,c)
    t, bs = ts, n
    # v1net_tile = tf.tile(X, [t,1,1,1])
    # v1net_input = tf.reshape(v1net_tile, [n,t,h,w,c])
    ones = tf.ones([t,1,1,1,1],dtype=tf.float32)
    v1net_input = ones*X
    v1net_input = tf.transpose(v1net_input, [1,0,2,3,4])
    state_c = tf.zeros([n,h,w,n_filt], dtype=X.dtype, name='init_c')
    state_h = tf.zeros([n,h,w,n_filt], dtype=X.dtype, name='init_h')
    #state_c = tf.zeros_like(X, name='init_c', dtype=tf.float32)
    #state_h = tf.zeros_like(X, name='init_h', dtype=tf.float32)
    state = tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h)
    # Following cell is V1Net w/ batch normalization per timestep
    v1net_cell = V1Net_BN_cell(
                            input_shape=[h,w,c],
                            output_channels=n_filt,
                            kernel_shape=[k_hw,k_hw],
                            inh_mult=inh_mult,
                            exc_mult=exc_mult,
                            activation=v1_act,
                            timesteps=t,
                            activation_hor=True,
                            pointwise=pointwise,
                            batchnorm=True,
                            training=training,
                            )
    v1net_out = tf.nn.dynamic_rnn(cell=v1net_cell,
                                inputs=v1net_input,
                                initial_state=state,
                                dtype=tf.float32,
                                )
    new_state_c, new_state_h = v1net_out[1]
    return new_state_h



def v1net_bn_layer_deprecated(X, ts, n_filt, k_hw, training, filter_init, filter_reg=None,
                     inh_mult=1.5, pointwise=False, exc_mult=3, v1_act='relu', name=None):
    """
    Implementing neurips_2020_7391_code layer with recurrent nonlinear
    horizontal inhib/exci interactions. This neurips_2020_7391_code block
    is state-initialized to zeros, and input for each timestep
    is the same as the feedforward drive during the first timestep.
    The output of neurips_2020_7391_code is batchnormalized before passing on to the next timestep.
    :param X: input for first/all timestep
    :param ts: number of timesteps for unrolling neurips_2020_7391_code
    :param n_filt: number of output channels
    :param k_hw: spatial dimensions of kernels
    :param training: boolean indicating train/test mode
    :param filter_init: name of initializer for weights and biases
    :param filter_reg: regularization for weights
    :param inh_mult: relative size of inhibitory connections
    :param exc_mult: relative size of excitatory connections
    :param v1_act: nonlinearity for neurips_2020_7391_code output
    :return: batch normalized activations of neurips_2020_7391_code post horizontal connection application
    """
    n,h,w,c = X.shape.as_list()
    print(n,h,w,c)
    t, bs = ts, n
    # trying hti neurips_2020_7391_code
    #ones = tf.concat((tf.ones([1,1,1,1,1]),tf.zeros([t-1,1,1,1,1])),axis=0)
    ones = tf.ones([t,1,1,1,1],dtype=X.dtype)
    v1net_input = ones*X
    v1net_input = tf.transpose(v1net_input, [1,0,2,3,4])
    state_c = tf.zeros([n,h,w,n_filt], dtype=X.dtype, name='init_c')
    state_h = tf.zeros([n,h,w,n_filt], dtype=X.dtype, name='init_h')
    # state_c = tf.zeros_like(X, name='init_c', dtype=X.dtype)
    # state_h = tf.zeros_like(X, name='init_h', dtype=X.dtype)
    state = tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h)
    v1net_cell = V1Net_cell(
                            input_shape=[h,w,c],
                            output_channels=n_filt,
                            kernel_shape=[k_hw,k_hw],
                            inh_mult=inh_mult,
                            exc_mult=exc_mult,
                            activation=v1_act,
                            timesteps=t,
                            pointwise=pointwise,
                            activation_hor=True,
                            )
    v1net_out = tf.nn.dynamic_rnn(cell=v1net_cell,
                                inputs=v1net_input,
                                initial_state=state,
                                dtype=X.dtype,
                                )
    new_state_c, new_state_h = v1net_out[1]
    new_state_h_bn = tf.layers.batch_normalization(
                            inputs=new_state_h,  # batch normalization of neurips_2020_7391_code's final output
                            axis=3,
                            center=True,
                            scale=True,
                            training=training,
                            fused=True,
                            gamma_initializer=tf.ones_initializer()
                            )
    return new_state_h_bn

def v1net_linear_layer(X, ts, n_filt, k_hw, filter_init, filter_reg,
                     inh_mult, exc_mult, v1_act, name=None):
    ## Horizontal inhibitory and
    ## excitatory connections
    ## influencing classical RF
    ## through V1net. Implementing
    ## correlation norm to enable
    ## like-tuned inhibition/excitation
    from layers.horizontal_cells.horizontal_lstm_linear_ei import Horizontal_Linear_EI

    n,h,w,c = X.shape.as_list()
    t,bs = ts, n
    v1net_tile = tf.tile(X, [t,1,1,1])
    v1net_input = tf.reshape(v1net_tile, [t,-1,h,w,c])
    ## TODO: Think of better initialization for neurips_2020_7391_code.
    state_c = tf.identity(X, name='init_c')
    state_h = tf.identity(X, name='init_h')
    state = tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h)
    # Trying linear horizontal interactions
    v1net_cell = Horizontal_Linear_EI(
                                input_shape=[h,w,c],
                                output_channels=n_filt,
                                kernel_shape=[k_hw,k_hw],
                                inh_mult=inh_mult,
                                exc_mult=exc_mult,
                                activation=v1_act,
                                batch_size=bs,
                                activation_hor=True,
                                timesteps=t,
                                )
    v1net_output = v1net_cell.call(
                        inputs=v1net_input,
                        state=state,
                        )
    (inputs, L_outputs,
     L_H_exc, L_H_inh, new_state) = v1net_output
    new_state_c, new_state_h = new_state
    return new_state_h


def cornets_layer(X, ts, n_filt, k_hw, training):
    """
    Function to implement CORNet-S
    :param X: input tensor
    :param ts: number of timesteps of recurrence
    :param n_filt: number of output channels
    :param k_hw: filter size
    :return: h, output state after recurrent computation
    """
    from layers.conv_rnn import conv_layers
    input_shape = X.shape
    with tf.variable_scope('cornet_s', reuse=tf.compat.v1.AUTO_REUSE):
        h = tf.keras.layers.Conv2D(filters=n_filt, kernel_size=1, padding='SAME',
                             use_bias=False, input_shape=input_shape, name='input_conv',
                             kernel_initializer=get_initializer('xavier'))(X)
        skipconv = tf.keras.layers.Conv2D(filters=n_filt, kernel_size=1,
                                          padding='SAME', use_bias=False,
                                          input_shape=input_shape, name='skip_conv',
                                          kernel_initializer=get_initializer('xavier'))
        conv1 = tf.keras.layers.Conv2D(filters=n_filt * 4, kernel_size=1, padding='SAME',
                                       use_bias=False, input_shape=input_shape, name='conv1',
                                       kernel_initializer=get_initializer('xavier'))
        conv2 = tf.keras.layers.Conv2D(filters=n_filt * 3, kernel_size=5, padding='SAME',
                                       use_bias=False, input_shape=input_shape, name='conv2',
                                       kernel_initializer=get_initializer('xavier'))
        conv3 = tf.keras.layers.Conv2D(filters=n_filt, kernel_size=1, padding='SAME',
                                       use_bias=False, input_shape=input_shape, name='conv3',
                                       kernel_initializer=get_initializer('xavier'))
        #bn3 = tf.keras.layers.BatchNormalization(name='bn3')
        for t in range(ts):
            if t is 0:
                skip = tf.compat.v1.layers.batch_normalization(skipconv(h), training=training)
            else:
                skip = h
            h = conv1(h); h = tf.compat.v1.layers.batch_normalization(h,training=training)
            h = tf.nn.relu(h)
            h = conv2(h); h = tf.compat.v1.layers.batch_normalization(h,training=training)
            h = tf.nn.relu(h)
            h = conv3(h); h = tf.compat.v1.layers.batch_normalization(h,training=training) 
            h = h + skip; h = tf.nn.relu(h)
    return h 
