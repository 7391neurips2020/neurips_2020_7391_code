import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt


def l1(y_hat, y, name):
    # Helper function to return
    # l1 loss for y_hat (prediction)
    # and y (label)
    l1 = tf.math.abs(tf.math.sub(y_hat, y))
    l1_mean = tf.math.reduce_mean(l1,name=name)
    return l1_mean


def l2(y_hat, y,name):
    # Helper function to return
    # l2 loss for y_hat (prediction)
    # and y (label)
    diff = tf.math.subtract(y_hat, y)
    l2 = tf.nn.l2_loss(diff)
    l2_mean = tf.math.reduce_mean(l2,name=name)
    return l2_mean


def bce(y_hat, y, name, weighted=False):
    # Helper function to return
    # Binary Cross Entropy
    # loss for y_hat (logits)
    # and y (labels)
    print('Adding BCE loss')
    if weighted:
        y_flat = tf.reshape(y,[-1,1])
        y_hat_flat = tf.reshape(y_hat,[-1,1])
        bce = tf.nn.weighted_cross_entropy_with_logits(
                                        targets=y_flat,
                                        logits=y_hat_flat,
                                        pos_weight=5,
                                        name=name,
                                        )
    else:
        bce = tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=y_hat,
                                    labels=y,
                                    name='BCE',
                                    )
    bce_mean = tf.math.reduce_mean(bce, name=name)
    return bce_mean


def cce(y_hat, y,name, num_classes=2):
    # Helper function to return
    # Categorical Cross Entropy
    # loss for y_hat (logits)
    # and y (labels)
    print('Adding CCE ctive')
    if len(y.shape)>2:
        y_flat = tf.reshape(y,[-1,num_classes])
        y_hat_flat = tf.reshape(y_hat, [-1,num_classes])
    else:
        y_flat = y
        y_hat_flat = y_hat
    cce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                logits=y_hat_flat,
                                labels=y_flat,
                                name=name,
                                )
    cce_mean = tf.math.reduce_mean(cce, name=name)
    return cce_mean


def correlation(y_hat, y, name, sigmoid=True):
    # Helper function to create
    # a correlation loss for edge
    # detection. Correlation loss
    # treats the learning problem as
    # maximizing the correlation between
    # the target distribution and prediction
    # distribution. Since loss is minimized,
    # we opt to negate the correlation and
    # minimize it to maximize label-prediction
    # correlation.
    y_flat = tf.reshape(y, [-1,1])
    y_hat_flat = tf.reshape(y_hat, [-1,1])
    if sigmoid:
        y_hat_flat = tf.math.sigmoid(y_hat_flat)
    y_mean_norm = y_flat - tf.reduce_mean(y_flat)
    y_hat_mean_norm = y_hat_flat - tf.reduce_mean(y_hat_flat)
    y_hat_mean_norm_t = tf.transpose(y_hat_mean_norm, (1,0))
    y_mean_norm_t = tf.transpose(y_mean_norm, (1,0))
    corr_num = tf.matmul(y_mean_norm_t, y_hat_mean_norm)
    corr_denom = tf.matmul(y_hat_mean_norm_t, y_hat_mean_norm) * \
                    tf.matmul(y_mean_norm_t,y_mean_norm)
    corr_denom = tf.sqrt(corr_denom)
    corr_loss = tf.divide(corr_num, corr_denom)
    corr_loss = tf.reduce_mean(corr_loss)
    corr_loss = 1.-corr_loss
    print('Adding correlation loss')
    return corr_loss


def add_loss(model_out, obj, labels):
    ## Function to add an appropriate
    ## loss function to our model
    if obj == 'l2':
        # Add an L2 loss
        L = l2(
                    model_out,
                    labels,
                    name='objective_min'
                    )
    if obj == 'l1':
        # Add an L1 loss
        L = l1(
                    model_out,
                    labels,
                    name='objective_min'
                    )
    if obj == 'bce':
        # Add binary cross entropy loss
        L = bce(
                    model_out,
                    labels,
                    name='objective_min',
                    )
    if obj == 'cce':
        # Add binary cross entropy loss
        L = cce(
                    model_out,
                    labels,
                    name='objective_min',
                    num_classes=10,
                    )
    if obj == 'correlation':
        L = correlation(
                    model_out,
                    labels,
                    name='objective_min',
                    )
    return L
