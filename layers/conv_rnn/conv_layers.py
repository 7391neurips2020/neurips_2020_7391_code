import numpy as np
import tensorflow as tf
from utils.tf_utils import *

def ff_layer(X, n_filt, k_hw, filter_init, padding='SAME', 
        filter_reg=None, name=None): #atrous=False):

    """Simple feedforward 2D convolution layer"""
    W_ret, B_ret = [], []
    weights_ret = get_filter(
                        in_shape=X.shape, #input shape
                        out_channels=n_filt,
                        ker_shape=k_hw,
                        filter_init=filter_init,
                        name=name, with_bias=True,
                        filter_reg=filter_reg,
                        )
    W_ret = weights_ret[0]

    if len(weights_ret)>1:
        B_ret = weights_ret[1]

    X = apply_conv(
                X, W_ret, padding,
                name='retina_conv',
                )
    X = apply_bias(
                X, B_ret,
                name='retina_out',
                )

    return X

def ff_layer_atrous(X, n_filt, k_hw, filter_init, 
                    padding='SAME', filter_reg=None, 
                    name=None):
    """Simple feedforward 2D convolution layer"""
    W_ret, B_ret = [], []
    weights_ret = get_filter(
                        in_shape=X.shape, #input shape
                        out_channels=n_filt,
                        ker_shape=k_hw,
                        filter_init=filter_init,
                        name=name, with_bias=True,
                        filter_reg=filter_reg,
                        )
    W_ret = weights_ret[0]

    if len(weights_ret)>1:
        B_ret = weights_ret[1]

    X = apply_atrous(
                X, W_ret, padding,
                name='retina_conv',
                )
    X = apply_bias(
                X, B_ret,
                name='retina_out',
                )

    return X

def alexnet_layer(X, ts, config, name=None):
    """Fill in AlexNet style retina"""
    pass

def vgg_layer(X, ts, config, name=None):
    """Fill in VGG style retina"""
    pass

def ff_layer_smcnn(X, n_filt=32):
    weights_ret = get_smcnn_kernel(in_shape=X.shape,
                                   out_channels=n_filt,
                                   ker_shape=2,
                                   )
    X = apply_conv(
                X, weights_ret, padding='SAME',
                )
    return X
