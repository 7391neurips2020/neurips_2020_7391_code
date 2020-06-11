import tensorflow as tf
from utils.tf_utils import *
from layers.conv_rnn import conv_layers
from layers.conv_rnn import rnn_layers
from layers.readouts import per_pixel_classification
import numpy as np

def nn_generator(images, labels, training, keep_prob, model_params, data_format='NHWC'):
    """
    Function to generate a neural network from dictionary of architecture params
    :param images: 4D input image tensor
    :param labels: ground truth tensor
    :param model_params: dictionary with architecture parameters
    :return: X, final layer output of neural network
    """
    model_name = list(model_params.keys())[0]
    model_arch = model_params[model_name]['arch']
    X = images
    with tf.variable_scope(model_name):
        for layer in model_arch:
            for layer_name, layer_params in layer.items():
                with tf.variable_scope(layer_name):
                    if layer_params['type']=='ff':
                        X = conv_layers.ff_layer(X, n_filt=layer_params['n_filt'],
                                                          k_hw=layer_params['k_hw'],
                                                          filter_init='xavier',
                                                          padding=layer_params['padding'],
                                                          name='conv',
                                                          )
                        # X = apply_act(X, layer_params['act'])
                        X = tf.layers.batch_normalization(
                                              inputs=X,
                                              axis=3,
                                              center=True,
                                              scale=True,
                                              training=training,
                                             fused=True,
                                              gamma_initializer=tf.ones_initializer())
                        X = apply_act(X, layer_params['act'])
                        # if not 'no_dropout' in layer_params.keys():
                        #    X = tf.compat.v1.nn.dropout(X, rate=1.-keep_prob)
                    elif layer_params['type'] == 'ff_smcnn':
                        X = conv_layers.ff_layer(X, n_filt=layer_params['n_filt'],
                                                          k_hw=layer_params['k_hw'],
                                                          filter_init='xavier',
                                                          padding=layer_params['padding'],
                                                          name='conv',
                                                          )
                        X = conv_layers.ff_layer_smcnn(X, n_filt=layer_params['n_filt'])
                        X = tf.layers.batch_normalization(
                                              inputs=X,
                                              axis=3,
                                              center=True,
                                              scale=True,
                                              training=training,
                                             fused=True,
                                              gamma_initializer=tf.ones_initializer())
                        X = apply_act(X, layer_params['act'])

                    elif layer_params['type'] == 'hgru':
                        X = rnn_layers.hgru_layer(X, ts=layer_params['ts'],
                                                        n_filt=layer_params['n_filt'],
                                                        k_hw=layer_params['k_hw'],
                                                        #pointwise=True,
                                                       name='hgru')
                        X = tf.layers.batch_normalization(inputs=X, axis=3, center=True,
							scale=True, training=training, fused=True,
							gamma_initializer=tf.ones_initializer())

                    elif layer_params['type']=='atrous':
                        X = conv_layers.ff_layer_atrous(X, n_filt=layer_params['n_filt'],
                                                        k_hw=layer_params['k_hw'],
                                                        filter_init='xavier',
                                                        padding=layer_params['padding'],
                                                        name='conv')
                        X = tf.layers.batch_normalization(
                                              inputs=X,
                                              axis=3,
                                              center=True,
                                              scale=True,
                                              training=training,
                                              fused=True,
                                              gamma_initializer=tf.ones_initializer())
                        X = apply_act(X, layer_params['act'])

                    elif layer_params['type']=='corblock_s':
                        X = rnn_layers.cornets_layer(X, ts=layer_params['ts'],
                                                    n_filt=layer_params['n_filt'],
                                                    k_hw=layer_params['k_hw'],
                                                    training=training,
                                                    )

                    elif layer_params['type']=='pool_avg':
                        pool_size = (X.shape[1], X.shape[2])
                        X = tf.layers.average_pooling2d(
                                        inputs=X, pool_size=pool_size, 
                                        strides=1, padding='VALID',
                                        )

                    elif layer_params['type']=='v1net_s':
                        X = rnn_layers.v1net_shnt_layer(X, ts=layer_params['ts'],
                                                        n_filt=layer_params['n_filt'],
                                                        k_hw=layer_params['k_hw'],
                                                        filter_init='xavier',
                                                        inh_mult=layer_params['inh_mult'],
                                                        exc_mult=layer_params['exc_mult'], 
                                                        v1_act=layer_params['act'],
                                                       name='v1net_shunt')
                        X = tf.layers.batch_normalization(
                                              inputs=X,
                                              axis=3,
                                              center=True,
                                              scale=True,
                                              training=training,
                                              fused=True,
                                              gamma_initializer=tf.ones_initializer()
                                              )
                        # X = tf.compat.v1.nn.dropout(X, rate=1.-keep_prob)

                    elif layer_params['type']=='v1net_s_bn':
                        X = rnn_layers.v1net_shunt_bn_layer(X, ts=layer_params['ts'],
                                                        n_filt=layer_params['n_filt'],
                                                        k_hw=layer_params['k_hw'],
                                                        filter_init='xavier',
                                                        inh_mult=layer_params['inh_mult'],
                                                        exc_mult=layer_params['exc_mult'], 
                                                        v1_act=layer_params['act'],
							training=training,
                                                        name='v1net_shunt_bn')
                        X = tf.layers.batch_normalization(inputs=X, training=training)

                    elif layer_params['type']=='lstm':
                        #TODO: Implement convlstm baseline layer
                        print('Adding LSTM layer')
                        X = rnn_layers.lstm_layer(X, ts=layer_params['ts'],
                                                n_filt=layer_params['n_filt'],
                                                k_hw=layer_params['k_hw'],
                                                name='v1net_lstm')

                    elif layer_params['type']=='pool2d':
                        X = apply_maxpool2d(X, k_hw=layer_params['k_hw'],
                                            strides=layer_params['strides'],
                                            name='pool2d')
                        if 'dropout' in layer_params.keys():
                            X = tf.compat.v1.nn.dropout(X, rate=1.-keep_prob)

                    elif layer_params['type']=='dropout':
                        X = tf.compat.v1.nn.dropout(X, rate=1.-keep_prob)

                    elif layer_params['type'] == 'dropout':
                        X = tf.compat.v1.nn.dropout(X, rate=1.-keep_prob)

                    elif layer_params['type']=='gap':
                        X = gap_readout(X, n_out=layer_params['n_units'],
                                        ker_shape=layer_params['k_hw'],
                                        W_init='xavier',
                                        name='gap_readout',
                                        )
                    elif layer_params['type']=='linear':
                        X = linear_readout(X, n_out=layer_params['n_units'],
                                        W_init='xavier',
                                        name='linear',
                                        )
                        X = apply_act(X, layer_params['act'])
                    elif layer_params['type']=='linear_bn':
                        X = linear_readout(X, n_out=layer_params['n_units'],
                                        W_init='xavier',
                                        name='linear_bn',
                                        )
                        X = tf.layers.batch_normalization(
                                              inputs=X,
                                              axis=-1,
                                              center=True,
                                              scale=True,
                                              training=training,
                                              fused=True,
                                              gamma_initializer=tf.ones_initializer())
                        X = apply_act(X, layer_params['act'])
                    elif layer_params['type']=='bsds':
                        n, h, w, c = X.shape
                        X, W, b = conv_readout_sigmoid(X, W_init='xavier',
                                                       ker_shape=1,
                                                       out_shape=[n,h,w,1],
                                                       name='bsds_readout',
                                                       return_weights=True,
                                                       )
                    else:
                        raise(ValueError('Layer type %s not implemented'%(layer_params['type'])))
            print('Added layer', layer, 'output:',X.shape)
    return X


def v1net_bn_shallow(images, labels, training, keep_prob=1., model_type='shallow_v1_bn', num_classes=2):
    model_params = {
            'shallow_v1_bn': {
                    'arch': [
                        {'retina/conv_1':
                            {
                                'type': 'ff',
                                'n_filt':32,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True,
                            },
                        },
                        {'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
                        {'v1/hor_conn_v1net':
                            {
                                'type': 'v1net_s_bn',
                                'n_filt': 64,
                                'k_hw': 5,
                                'ts': 5,
                                'inh_mult': 1.5,
                                'exc_mult': 3,
                                'padding': 'VALID',
                                'act': 'relu'
                            },
                        },
                        {'v1/pool1':
                            {
                                'type': 'pool_avg',
                                #'type': 'pool2d',
                                #'k_hw': [1,2,2,1],
                                #'strides': [1,2,2,1],
                                #'dropout': True,
                            }
                        },
                        {'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ]
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params


def hgru_shallow(images, labels, training, keep_prob=1., model_type='shallow_hgru', num_classes=2):
    model_params = {
            'shallow_hgru': {
                    'arch': [ 
                        {'retina/conv_1':
                            {
                                'type': 'ff',
                                'n_filt':32,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True,
                            },
                        },
                        {'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
                        {'v1/hor_conn_v1net':
                            {
                                'type': 'hgru',
                                'n_filt': 32,
                                'k_hw': 5,
                                'ts': 5,
                                'inh_mult': 1.5,
                                'exc_mult': 3,
                                'padding': 'VALID',
                                'act': 'relu'
                            },
                        },
                        {'v1/pool1':
                            {
                                'type': 'pool_avg',
                            }
                        },
                        {'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ]
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params

def ff_7l_smcnn(images, labels, training, keep_prob=1., model_type='ff_7l_smcnn', num_classes=2):
    model_params = {
            'ff_7l_smcnn': {
                    'arch': [
                        {'retina/conv_1':
                            {
                                'type': 'ff_smcnn',
                                'n_filt':32,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True
                            },
                        },
			{'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
			{'v1/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v1/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
			{'v1/conv_3':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v2/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v2/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v1/pool1':
                            {
                                'type': 'pool_avg',
                            }
                        },
			{'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ],
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params


def cornets_shallow(images, labels, training, keep_prob=1., model_type='shallow_cornets', num_classes=2):
    model_params = {
            'shallow_cornets': {
                    'arch': [ 
                        {'retina/conv_1':
                            {
                                'type': 'ff',
                                'n_filt':32,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True,
                            },
                        },
                        {'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
                        {'v1/hor_conn_cornets':
                            {
                                'type': 'corblock_s',
                                'n_filt': 32,
                                'k_hw': 5,
                                'ts': 5,
                                'padding': 'VALID',
                                'act': 'relu'
                            },
                        },
                        {'v1/pool1':
                            {
                                #'type': 'pool2d',
                                'type': 'pool_avg',
                                #'k_hw': [1, 2, 2, 1],
                                #'strides': [1, 2, 2, 1],
                            }
                        },
                        {'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ]
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params


def v1net_shallow(images, labels, training, keep_prob=1., model_type='shallow_v1', num_classes=2):
    model_params = {
            'shallow_v1': {
                    'arch': [ 
                        {'retina/conv_1':
                            {
                                'type': 'ff',
                                'n_filt':32,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True,
                            },
                        },
                        #{'retina/conv_2':
                        #    {
                        #        'type': 'ff',
                        #        'n_filt':32,
                        #        'k_hw': 5,
                        #        'padding': 'SAME',
                        #        'act': 'relu',
                        #        'no_dropout': True,
                        #    },
                        #},
                        {'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                        #        'dropout': True,
                            }
                        },
                        {'v1/hor_conn_v1net':
                            {
                                'type': 'v1net_s',
                                'n_filt': 32,
                                'k_hw': 5,
                                'ts': 5, #timesteps,
                                'inh_mult': 1.5,
                                'exc_mult': 3,
                                'padding': 'VALID',
                                'act': 'relu'
                            },
                        },
                        {'v1/pool1':
                            {
                                #'type': 'pool2d',
                                'type': 'pool_avg',
                                #'k_hw': [1, 2, 2, 1],
                                #'strides': [1, 2, 2, 1],
                            #    'dropout': True
                            }
                        },
                        {'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ]
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params

def ff_shallow(images, labels, training, keep_prob=1., model_type='ff_shallow', num_classes=2):
    model_params = {
            'ff_shallow': {
                    'arch': [
                        {'retina/conv_1':
                            {
                                'type': 'ff',
                                'n_filt':32,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True
                            },
                        },
			{'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
			{'v1/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
			{'v1/pool1':
                            {
                                'type': 'pool_avg',
                                # 'k_hw': [1, 2, 2, 1],
                                # 'strides': [1, 2, 2, 1],
                            }
                        },
			{'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                # 'k_hw': 3,
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ],
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params

def ff_shallow_atrous(images, labels, training, keep_prob=1., model_type='ff_shallow_atrous', num_classes=2):
    model_params = {
            'ff_shallow_atrous': {
                    'arch': [
                        {'retina/conv_1':
                            {
                                'type': 'ff',
                                'n_filt':32,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True
                            },
                        },
			{'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
			{'v1/conv_1':
                            {
                                'type': 'atrous',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
			{'v1/pool1':
                            {
                                'type': 'pool_avg',
                                # 'k_hw': [1, 2, 2, 1],
                                # 'strides': [1, 2, 2, 1],
                            }
                        },
			{'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                # 'k_hw': 3,
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ],
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params

def ff_4l_atrous(images, labels, training, keep_prob=1., model_type='ff_4l', num_classes=2):
    model_params = {
            'fft_4l': {
                    'arch': [
                        {'retina/conv_1':
                            {
                                'type': 'atrous',
                                'n_filt':32,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True
                            },
                        },
			{'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
			{'v1/conv_1':
                            {
                                'type': 'atrous',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v1/conv_2':
                            {
                                'type': 'atrous',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
			{'v1/conv_3':
                            {
                                'type': 'atrous',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
			{'v1/pool1':
                            {
                                'type': 'pool_avg',
                                # 'k_hw': [1, 2, 2, 1],
                                # 'strides': [1, 2, 2, 1],
                            }
                        },
			{'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                # 'k_hw': 3,
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ],
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params

def ff_4l(images, labels, training, keep_prob=1., model_type='ff_4l', num_classes=2):
    model_params = {
            'fft_4l': {
                    'arch': [
                        {'retina/conv_1':
                            {
                                'type': 'ff',
                                'n_filt':32,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True
                            },
                        },
			{'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
			{'v1/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v1/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
			{'v1/conv_3':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
			{'v1/pool1':
                            {
                                'type': 'pool_avg',
                                # 'k_hw': [1, 2, 2, 1],
                                # 'strides': [1, 2, 2, 1],
                            }
                        },
			{'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                # 'k_hw': 3,
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ],
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params

def ff_7l_atrous(images, labels, training, keep_prob=1., model_type='ff_7l_atrous', num_classes=2):
    model_params = {
            'ff_7l': {
                    'arch': [
                        {'retina/conv_1':
                            {
                                'type': 'atrous',
                                'n_filt':32,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True
                            },
                        },
			{'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
			{'v1/conv_1':
                            {
                                'type': 'atrous',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v1/conv_2':
                            {
                                'type': 'atrous',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
			{'v1/conv_3':
                            {
                                'type': 'atrous',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v2/conv_1':
                            {
                                'type': 'atrous',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v2/conv_2':
                            {
                                'type': 'atrous',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v1/pool1':
                            {
                                'type': 'pool_avg',
                            }
                        },
			{'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ],
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params

def ff_7l_large(images, labels, training, keep_prob=1., model_type='ff_7l', num_classes=2):
    model_params = {
            'ff_7l': {
                    'arch': [
                        {'retina/conv_1':
                            {
                                'type': 'ff',
                                'n_filt':32,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True
                            },
                        },
		        {'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
		        {'v1/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 10,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                        },
                        {'v1/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 10,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                        },
		        {'v1/conv_3':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 10,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                        },
         #{'v1/pool1':
                        #    {
                        #        'type': 'pool2d',
                        #        'k_hw': [1, 2, 2, 1],
                        #        'strides': [1, 2, 2, 1],
                        #    }
                        #},
                        {'v2/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 10,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                        },
                        {'v2/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 10,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                        },
                        #{'v2/conv_3':
                        #    {
                        #        'type': 'ff',
                        #        'n_filt': 128,
                        #        'k_hw': 5,
                        #        'padding': 'VALID',
                        #        'act': 'relu'
                        #    }
                        #},
                        {'v1/pool1':
                            {
                                'type': 'pool_avg',
                                # 'k_hw': [1, 2, 2, 1],
                                # 'strides': [1, 2, 2, 1],
                            }
                        },
		        {'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                # 'k_hw': 3,
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ],
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params, data_format='NCHW'), model_params

def ff_7l_same(images, labels, training, keep_prob=1., model_type='ff_7l_same', num_classes=2):
    model_params = {
            'ff_7l_same': {
                    'arch': [
                        {'retina/conv_1':
                            {
                                'type': 'ff',
                                'n_filt':32,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True
                            },
                        },
			{'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
			{'v1/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v1/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
			{'v1/conv_3':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v2/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v2/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v1/pool1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                                'dropout': True,
                            }
                        },
			{'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ],
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params

def ff_7l(images, labels, training, keep_prob=1., model_type='ff_7l', num_classes=2):
    model_params = {
            'ff_7l': {
                    'arch': [
                        {'retina/conv_1':
                            {
                                'type': 'ff',
                                'n_filt':32,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True
                            },
                        },
			{'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
			{'v1/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v1/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
			{'v1/conv_3':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
			#{'v1/pool1':
                        #    {
                        #        'type': 'pool2d',
                        #        'k_hw': [1, 2, 2, 1],
                        #        'strides': [1, 2, 2, 1],
                        #    }
                        #},
                        {'v2/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v2/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        #{'v2/conv_3':
                        #    {
                        #        'type': 'ff',
                        #        'n_filt': 128,
                        #        'k_hw': 5,
                        #        'padding': 'VALID',
                        #        'act': 'relu'
                        #    }
                        #},
                        {'v1/pool1':
                            {
                                'type': 'pool_avg',
                                # 'k_hw': [1, 2, 2, 1],
                                # 'strides': [1, 2, 2, 1],
                            }
                        },
			{'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                # 'k_hw': 3,
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ],
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params

def ff_9l(images, labels, training, keep_prob=1., model_type='ff_9l', num_classes=2):
    model_params = {
            'ff_9l': {
                    'arch': [
                        {'retina/conv_1':
                            {
                                'type': 'ff',
                                'n_filt':32,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True
                            },
                        },
			{'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
			{'v1/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 64,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v1/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 64,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
			{'v1/conv_3':
                            {
                                'type': 'ff',
                                'n_filt': 64,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
			#{'v1/pool1':
                        #    {
                        #        'type': 'pool2d',
                        #        'k_hw': [1, 2, 2, 1],
                        #        'strides': [1, 2, 2, 1],
                        #    }
                        #},
                        {'v2/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 64,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v2/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 64,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v1/pool1':
                            {
                                'type': 'pool_avg',
                                # 'k_hw': [1, 2, 2, 1],
                                # 'strides': [1, 2, 2, 1],
                            }
                        },
			{'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                # 'k_hw': 3,
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ],
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params


def ff_9l_atrous(images, labels, training, keep_prob=1., model_type='ff_9l_atrous', num_classes=2):
    model_params = {
            'ff_9l_atrous': {
                    'arch': [
                        {'retina/conv_1':
                            {
                                'type': 'atrous',
                                'n_filt':32,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True
                            },
                        },
			{'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
			{'v1/conv_1':
                            {
                                'type': 'atrous',
                                'n_filt': 64,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v1/conv_2':
                            {
                                'type': 'atrous',
                                'n_filt': 64,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
			{'v1/conv_3':
                            {
                                'type': 'atrous',
                                'n_filt': 64,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v2/conv_1':
                            {
                                'type': 'atrous',
                                'n_filt': 64,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v2/conv_2':
                            {
                                'type': 'atrous',
                                'n_filt': 64,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                        },
                        {'v1/pool1':
                            {
                                'type': 'pool_avg',
                            }
                        },
			{'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ],
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params


def v1net_nl(images, labels, training, keep_prob=1., model_type='v1net_4l', num_classes=2):
    model_params = {
            'v1net_4l': {
                    'arch': [
                        {'retina/conv_1':
                            {
                                'type': 'ff',
                                'n_filt':16,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True
                            },
                        },
			{'retina/hor_conn_v1net':
                            {
                                # 'type': 'ff',
                                # 'type': 'v1net_s',
                                'type': 'lstm',
                                'n_filt': 16,
                                'k_hw': 5,
                                'ts': 4,
                                'inh_mult': 1.5,
                                'exc_mult': 3,
                                # 'padding': 'VALID',
                                'act': 'relu',
                                'no_dropout': True
                            },
                        },
                        {'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
                        {'v1/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                        },
			{'v1/conv_2':
                                {
                                    'type': 'ff',
                                    'n_filt': 64,
                                    'k_hw': 5,
                                    'padding': 'VALID',
                                    'act': 'relu'
                                }
                         },
                        {'v1/pool1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
                        {'v4/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 128,
                                'k_hw': 3,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                        },
                        {'v4/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 128,
                                'k_hw': 3,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                        },
                        {'v4/pool1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
                        {'v5/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 256,
                                'k_hw': 3,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                        },
                        {'v5/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 256,
                                'k_hw': 3,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                        },
                        {'v5/pool1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
                        {'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                # 'k_hw': 3,
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ],
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params

def v1net_middle_nl(images, labels, training, keep_prob=1., model_type='v1net_4l', num_classes=2):
    model_params = {
            'v1net_4l': {
                    'arch': [
                        {'retina/conv_1':
                            {
                                'type': 'ff',
                                'n_filt':32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True
                            },
                        },
                        {'retina/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 32,
                                'k_hw': 5,
                                'padding': 'SAME',
                                'act': 'relu',
                                'no_dropout': True
                            },
                        },
                        {'retina/pool_1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
                        {'v1/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 64,
                                'k_hw': 5,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                        },
                        {'v1/hor_conn_v1net':
                                {
                                    'type': 'v1net_s',
                                    'n_filt': 64,
                                    'k_hw': 5,
				    'inh_mult': 1.5,
				    'exc_mult': 3,
                                    'ts': 4,
                                    'padding': 'VALID',
                                    'act': 'relu'
                                }
                         },
                        {'v1/pool1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
                        {'v4/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 128,
                                'k_hw': 3,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                        },
                        {'v4/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 128,
                                'k_hw': 3,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                        },
                        {'v4/pool1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
                        {'v5/conv_1':
                            {
                                'type': 'ff',
                                'n_filt': 256,
                                'k_hw': 3,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                        },
                        {'v5/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 256,
                                'k_hw': 3,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                        },
                        {'v5/pool1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                        },
                        {'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                        },
                        {'readout':
                            {
                                'type': 'linear',
                                # 'k_hw': 3,
                                'n_units': num_classes,
                                'act': None,
                            },
                        },
                    ],
                    }
            }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params


def ff_binary(images, labels, training, keep_prob=1., model_type='ff_binary', num_classes=2):
    model_params = {
            'ff_binary': {
                    'arch': [
                            {'retina/conv_1':
                                {
                                    'type': 'ff',
                                    'n_filt':32,
                                    'k_hw': 5,
                                    'padding': 'SAME',
                                    'act': 'relu',
                                }
                            },
                            {'retina/conv_2':
                                {
                                    'type': 'ff',
                                    'n_filt': 32,
                                    'k_hw': 5,
                                    'padding': 'VALID',
                                    'act': 'relu',
                                }
                            },
                            {'retina/pool_1':
                                {
                                    'type': 'pool2d',
                                    'k_hw': [1,2,2,1],
                                    'strides': [1,2,2,1],
                                }
                            },
                            {'v1/conv_1':
                                {
                                    'type': 'ff',
                                    'n_filt': 64,
                                    'k_hw': 5,
                                    'padding': 'VALID',
                                    'act': 'relu'
                                }
                            },
                            {'v1/conv_2':
                                {
                                    'type': 'ff',
                                    'n_filt': 64,
                                    'k_hw': 5,
                                    'padding': 'VALID',
                                    'act': 'relu'
                                }
                            },
                            {'v1/pool1':
                                {
                                    'type': 'pool2d',
                                    'k_hw': [1,2,2,1],
                                    'strides': [1,2,2,1],
                                }
                            },
                            {'v4/conv_1':
                                {
                                'type': 'ff',
                                'n_filt': 128,
                                'k_hw': 3,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                            },
                            {'v4/conv_2':
                            {
                                'type': 'ff',
                                'n_filt': 128,
                                'k_hw': 3,
                                'padding': 'VALID',
                                'act': 'relu'
                            }
                            },
                            {'v4/pool1':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                            }
                            },
                            {'v5/conv_1':
                                {
                                    'type': 'ff',
                                    'n_filt': 256,
                                    'k_hw': 3,
                                    'padding': 'VALID',
                                    'act': 'relu'
                                }
                            },
                            {'v5/conv_2':
                                {
                                    'type': 'ff',
                                    'n_filt': 256,
                                    'k_hw': 3,
                                    'padding': 'VALID',
                                    'act': 'relu'
                                }
                            },
                            {'v5/pool1':
                                {
                                    'type': 'pool2d',
                                    'k_hw': [1, 2, 2, 1],
                                    'strides': [1, 2, 2, 1],
                                }
                            },
                            {'v5/linear':
                            {
                                'type': 'linear',
                                'n_units': 512 ,
                                'act': 'relu',
                            }
                            },
                            {'readout':
                                {
                                    # TODO: Add activation for gap readout
                                    'type': 'linear',
                                    'n_units': num_classes,
                                    'act': None,
                                }
                            },
                        ]
                    },
                }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params


def vgg16(images, labels, training, keep_prob=1., model_type='vgg16', num_classes=2):
    model_params = {
            'vgg16': {
                    'arch': [
                            {'conv_1/conv1_1':
                                {
                                    'type': 'ff',
                                    'n_filt':64,
                                    'k_hw': 3,
                                    'padding': 'SAME',
                                    'act': 'relu',
                                }
                            },
                            {'conv_1/conv1_2':
                                {
                                    'type': 'ff',
                                    'n_filt': 64,
                                    'k_hw': 3,
                                    'padding': 'SAME',
                                    'act': 'relu',
                                }
                            },
                            {'pool_1':
                                {
                                    'type': 'pool2d',
                                    'k_hw': [1,2,2,1],
                                    'strides': [1,2,2,1],
                                    'dropout': True,
                                }
                            },
                            {'conv_2/conv2_1':
                                {
                                    'type': 'ff',
                                    'n_filt': 128,
                                    'k_hw': 3,
                                    'padding': 'SAME',
                                    'act': 'relu'
                                }
                            },
                            {'conv_2/conv2_2':
                                {
                                    'type': 'ff',
                                    'n_filt': 128,
                                    'k_hw': 3,
                                    'padding': 'SAME',
                                    'act': 'relu'
                                }
                            },
                            {'pool2':
                                {
                                    'type': 'pool2d',
                                    'k_hw': [1,2,2,1],
                                    'strides': [1,2,2,1],
                                    'dropout': True,
                                }
                            },
                            {'conv_3/conv3_1':
                                {
                                'type': 'ff',
                                'n_filt': 256,
                                'k_hw': 3,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                            },
                            {'conv_3/conv3_2':
                            {
                                'type': 'ff',
                                'n_filt': 256,
                                'k_hw': 3,
                                'padding': 'SAME',
                                'act': 'relu'
                            }
                            },
                            {'conv_3/conv3_3':
                                {
                                    'type': 'ff',
                                    'n_filt': 256,
                                    'k_hw': 3,
                                    'padding': 'SAME',
                                    'act': 'relu'
                                }
                            },
                            {'pool3':
                            {
                                'type': 'pool2d',
                                'k_hw': [1, 2, 2, 1],
                                'strides': [1, 2, 2, 1],
                                'dropout': True,
                            }
                            },
                            {'conv_4/conv4_1':
                                {
                                    'type': 'ff',
                                    'n_filt': 512,
                                    'k_hw': 3,
                                    'padding': 'SAME',
                                    'act': 'relu'
                                }
                            },
                            {'conv_4/conv4_2':
                                {
                                    'type': 'ff',
                                    'n_filt': 512,
                                    'k_hw': 3,
                                    'padding': 'SAME',
                                    'act': 'relu'
                                }
                            },
                            {'conv_4/conv4_3':
                                {
                                    'type': 'ff',
                                    'n_filt': 512,
                                    'k_hw': 3,
                                    'padding': 'SAME',
                                    'act': 'relu'
                                }
                            },
                            {'pool4':
                                {
                                    'type': 'pool2d',
                                    'k_hw': [1, 2, 2, 1],
                                    'strides': [1, 2, 2, 1],
                                    'dropout': True,
                                }
                            },
                            {'conv_5/conv5_1':
                                {
                                    'type': 'ff',
                                    'n_filt': 512,
                                    'k_hw': 3,
                                    'padding': 'SAME',
                                    'act': 'relu'
                                }
                            },
                            {'conv_5/conv5_2':
                                {
                                    'type': 'ff',
                                    'n_filt': 512,
                                    'k_hw': 3,
                                    'padding': 'SAME',
                                    'act': 'relu'
                                }
                            },
                            {'conv_5/conv5_3':
                                {
                                    'type': 'ff',
                                    'n_filt': 512,
                                    'k_hw': 3,
                                    'padding': 'SAME',
                                    'act': 'relu'
                                }
                            },
                            {'pool5':
                                {
                                    'type': 'pool2d',
                                    'k_hw': [1, 2, 2, 1],
                                    'strides': [1, 2, 2, 1],
                                    'dropout': True,
                                }
                            },
                            {'fc6/conv':
                            {
                                'type': 'ff',
                                'n_filt': 4096,
                                'k_hw': 7,
                                'padding': 'SAME',
                                'act': 'relu',
                            }
                            },
                            {'fc6/dropout':
                                {
                                    'type': 'dropout',
                                }
                            },
                            {'fc7/conv':
                                {
                                    'type': 'ff',
                                    'n_filt': 4096,
                                    'k_hw': 1,
                                    'padding': 'SAME',
                                    'act': 'relu',
                                }
                            },
                            {'global_pool/pool_avg':
                                {
                                    'type': 'pool_avg',
                                }
                            },
                            {'readout':
                                {
                                    # TODO: Add activation for gap readout
                                    'type': 'linear',
                                    'n_units': num_classes,
                                    'act': None,
                                }
                            },
                        ]
                    },
                }
    return nn_generator(images, labels, training, keep_prob, model_params), model_params
