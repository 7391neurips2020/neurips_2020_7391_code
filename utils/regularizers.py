import tensorflow as tf
import numpy as np

def return_l2_reg(weights):
    """Function to apply l2 regularization
    to `weights`, returns a scalar
    :param weights: weights tensor to which
    l2 regularization is applied
    l2(weights) = sum(||w||^2)"""
    l2 = tf.math.square(weights)
    l2 = tf.math.reduce_sum(l2)
    return l2

def return_l1_reg(weights):
    """Function to apply l1 regularization
    to `weights`, returns a scalar
    :param weights: weights tensor to which
    l1 regularization is applied
    l1(weights) = sum(||w||^1)"""
    l1 = tf.math.abs(weights)
    l1 = tf.math.reduce_sum(l1)
    return l1

def return_horizontal_norm(Z):
    """Function to return the horizontal
    norm from activities. By maximizing
    trace of correlation matrix, each feature
    is maximally influenced only by itself.
    :param Z: Activities resulting from horizontal
    connections.
    corr = Z'.Z; hor_norm(Z) = trace(corr);
    """
    n,h,w,c = Z.shape
    z_flat = tf.reshape([-1,c])
    z_transpose = tf.transpose(z_flat, [1,0])
    corr = tf.matmul(z_transpose, z_flat)
    # Return the diagonal elements from
    # correlation matrix
    trace = tf.matrix_diag_part(corr)
    trace = tf.math.reduce_mean(trace)
    return trace
