import numpy as np
import tensorflow as tf
from utils.file_utils import *

# All tensorflow utilities bundled
# in this script


def compute_niters(ns, ne, bs):
    """Function to compute number of
    iterations using size of dataset,
    batch size, and the number of epochs
    specified"""
    num_iters = int(ns * ne / bs)
    return num_iters


def return_identity(x, y):
    return tf.identity(x), tf.identity(y)


def augment_images(image, label):
    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    import math
    print('Augmenting images..')
    distorted_image = tf.image.random_brightness(image,
                                                 max_delta=0.3)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.4, upper=1.8)
    distorted_image = tf.image.random_hue(distorted_image, 0.08)
    distorted_image = tf.image.random_saturation(distorted_image, 0.5, 1.)
    theta = tf.random.uniform(shape=[], minval=-np.pi / 8, maxval=np.pi / 8)
    distorted_image = tf.contrib.image.transform(
        distorted_image,
        tf.contrib.image.angles_to_projective_transforms(
            theta, tf.cast(tf.shape(image)[1], tf.float32),
            tf.cast(tf.shape(image)[2], tf.float32),
        )
    )
    distorted_label = tf.contrib.image.transform(
        label,
        tf.contrib.image.angles_to_projective_transforms(
            theta, tf.cast(tf.shape(label)[1], tf.float32),
            tf.cast(tf.shape(label)[2], tf.float32)
        )
    )
    distorted_image = tf.reshape(distorted_image, [-1, 321, 481, 3])
    distorted_label = tf.reshape(distorted_label, [-1, 321, 481, 1])
    print('Augmented using 6 techniques')
    # # Subtract off the mean and divide by the variance of the pixels.
    return distorted_image, distorted_label


def smcnn_kernel(k=2, sig_e=1.2, sig_i=1.8):
    """
    Function to create an SMCNN kernel (DoG surround modulation)
    :param k: length of DoG
    :param sig_e: excitation variance of DoG
    :param sig_i: inhibition variance of DoG
    """

    def gkern(l=5, sig=1.):
        """\
        creates gaussian kernel with side length l and a sigma of sig
        """
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
        return kernel / np.sum(kernel)

    g_e = gkern(2 * k + 1, sig_e)
    g_i = gkern(2 * k + 1, sig_i)
    g_sm = g_e - g_i
    g_sm /= g_sm[k, k]
    g_sm = np.round(g_sm, 3)
    g_sm = np.float32(g_sm)
    return g_sm


def ones_kernel(k=2):
    kern = np.ones((2 * k + 1, 2 * k + 1))
    kern /= np.sum(kern)
    return kern


def get_smcnn_kernel(in_shape, out_channels, ker_shape=2):
    n_in = in_shape[-1]
    n_out = out_channels
    k = ker_shape
    k_hw = 2 * k + 1
    n_out_half = int(n_out / 2)
    smkern = smcnn_kernel(k)
    onekern = ones_kernel(k)
    smkern_half = np.array([smkern] * n_in * n_out_half).reshape((n_in, n_out_half, k_hw, k_hw)).transpose((2, 3, 0, 1))
    onekern_half = np.array([onekern] * n_in * n_out_half).reshape((n_in, n_out_half, k_hw, k_hw)).transpose((2, 3, 0, 1))
    smcnn_full = np.concatenate([smkern_half, onekern_half], -1)
    smcnn_full = np.float32(smcnn_full)
    smcnn = tf.Variable(initial_value=smcnn_full,
                            dtype=tf.float32,name='smcnn_kernel_fixed',
                            trainable=False)
    return smcnn


def get_filter(in_shape, out_channels,
              ker_shape, filter_init,
              loc=0.0, scale=1e-3,
              with_bias=True,
              filter_reg=None,
              name=None, data_format='NHWC',
              ):
    # Function to obtain a conv kernel
    # with the specified shape and initialization
    if data_format=='NHWC':
        n, h, w, in_channels = in_shape
    else:
        n, in_channels, h, w = in_shape
    initializer = get_initializer(filter_init)
    # regularizer = get_regularizer(filter_reg)
    w_shape = [ker_shape, ker_shape, in_channels, out_channels]
    W = tf.get_variable(
                        name='%s_W'%(name),
                        shape=w_shape,
                        dtype=tf.float32,
                        # regularizer=regularizer,
                        initializer=initializer,
                        )
    if with_bias:
        b = tf.get_variable(
                        name='%s_b'%(name),
                        shape=[out_channels],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                        )
        return [W, b]
    return [W]


def get_W_b(in_shape, n_out,
            W_init, W_reg=None,
            loc=0.0, scale=1e-3,
            name=None,
            ):
    # Function to return W and b
    # for a linear readout layer
    n, n_in = in_shape
    initializer = get_initializer(W_init)
    regularizer = get_regularizer(W_reg)
    W = tf.get_variable(
                        name='%s_W'%(name),
                        shape=[n_in, n_out],
                        dtype=tf.float32,
                        regularizer=regularizer,
                        initializer=initializer,
                        )
    b = tf.get_variable(
                        name='%s_b'%(name),
                        shape=[n_out],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                        )
    return [W, b]


def get_regularizer(filter_reg):
    # Function to return a
    # weight regularizer based
    # on filter_reg
    print('Filter_reg',filter_reg)
    if filter_reg is None:
        return None
    if filter_reg=="l1":
        regularizer = tf.keras.regularizers.l1(l=0.01)
    if filter_reg=="l2":
        regularizer = tf.keras.regularizers.l2(l=0.01)
    # TODO: Implement more regularizers
    return regularizer


def get_initializer(filter_init):
    # Function to return a random
    # weight initializer
    # based on filter_init
    if filter_init=='xavier':
        initializer = tf.compat.v1.variance_scaling_initializer(
                            seed=None,
                            dtype=tf.float32,
                            )
        return initializer
    if filter_init=='trunc_norm':
        init_func = tf.initializers.truncated_normal
    if filter_init=='norm':
        init_func = tf.initializers.random_normal
    if filter_init=='uniform':
        init_func = tf.initializers.random_uniform
    initializer = init_func(
                            mean=loc,
                            stddev=scale,
                            seed=None,
                            dtype=tf.float32,
                            )
    return initializer


def normalize_input(X, name=None, norm_range=[0,1]):
    # Function to normalize input
    # to the given input range
    if norm_range == [-1,1]:
        X -= tf.constant(127.5)
        X /= tf.constant(127.5)
    if norm_range == [0,1]:
        X /= tf.constant(255.)
    return X


def apply_bias(X, b, name=None):
    ## Function to apply bias to
    ## incoming input
    X = tf.nn.bias_add(X, b, name=name)
    return X

def apply_atrous(X, W, padding='SAME', name=None):
    X = tf.compat.v1.nn.atrous_conv2d(
                X, W,
                rate=2,
                padding=padding,
                name=name)
    return X

def apply_conv(X, W, padding='SAME', name=None):
    ## Function to convolve X with W
    X = tf.compat.v1.nn.conv2d(
                    X, W,
                    strides=[1,1,1,1],
                    padding=padding,
                    name=name,
                    )
    return X


def apply_atrous_conv(X, W, name=None):
    ## Function to convolve X with W
    X = tf.nn.atrous_conv2d(
                    X, W,
                    padding='SAME',
                    rate=2,
                    name=name,
                    )
    return X


def apply_maxpool2d(X, k_hw=2, strides=2, name=None):
    # Function to apply 2d max pool
    X = tf.nn.max_pool(X, ksize=k_hw,
                       strides=strides,
                       padding='SAME',
                       name=name,
                       )
    return X


def apply_act(X, act, name=None):
    # Function to apply activation
    # function act, to input X
    if act == 'sigmoid':
        X = tf.nn.sigmoid(X, name='sig_out')
    if act == 'tanh':
        X = tf.nn.tanh(X, name='tanh_out')
    if act == 'relu':
        X = tf.nn.relu(X, name='relu_out')
    if act == 'selu':
        X = tf.nn.selu(X, name='selu_out')
    return X


def apply_norm(X, training):
    # Function to apply normalization to input
    X = tf.layers.batch_normalization(
        inputs=X,
        axis=3,
        center=True,
        scale=True,
        training=training,
        fused=True,
        gamma_initializer=tf.ones_initializer())

    return X


def conv_readout_sigmoid(X, out_shape,
                W_init, ker_shape,
                W_reg=None,
                name=None,
                return_weights=False
                ):
    n, h, w, c = X.shape
    n_out = out_shape[-1]
    weights_readout = get_filter(
                        in_shape=X.shape,
                        out_channels=n_out,
                        ker_shape=ker_shape,
                        filter_init=W_init,
                        filter_reg=W_reg,
                        name=name,
                        )
    W_readout = weights_readout[0]
    if len(weights_readout)>1:
        B_readout = weights_readout[1]
    X = apply_conv(
                X, W_readout,
                name='readout_conv'
                )
    X = apply_bias(
                X, B_readout,
                name='readout_out'
                )
    if return_weights:
        return X, W_readout, B_readout
    return X


def conv_readout_softmax(X, out_shape,
                W_init, ker_shape,
                W_reg=None,
                name=None,
                ):
    n, h, w, c = X.shape
    n_out = out_shape[-1]
    weights_readout = get_filter(
                        in_shape=X.shape,
                        out_channels=n_out,
                        ker_shape=ker_shape,
                        filter_init=W_init,
                        filter_reg=W_reg,
                        name=name,
                        )
    W_readout = weights_readout[0]
    if len(weights_readout)>1:
        B_readout = weights_readout[1]
    X = apply_conv(
                X, W_readout,
                name='readout_conv'
                )
    X = apply_bias(
                X, B_readout,
                name='readout_out'
                )
    return X


def linear_readout(X, n_out,
                  W_init, W_reg=None,
                  name=None):
    ## Function to attach the readout layer to
    ## Re2V1 circuit.
    if len(X.shape)>2:
        n, h, w, c = X.shape
        n_in = h*w*c # Flattening x into [n,-1]
        flattened_X = tf.reshape(X, [n,-1])
    else:
        flattened_X = tf.identity(X)
    weights_readout = get_W_b(
                        in_shape=flattened_X.shape,
                        n_out=n_out,
                        W_init=W_init,
                        W_reg=W_reg,
                        name='readout',
                        )
    W_readout, B_readout = weights_readout[0], weights_readout[1]
    X = tf.nn.xw_plus_b(
                        flattened_X, W_readout,
                        B_readout,
                        name=name
                        )
    return X


def gap_readout(X, n_out,
            W_init, ker_shape,
            W_reg=None,name=None,
            ):
    ## Function to attach a 1x1 convolution
    ## followed by Global average pooling
    print('Applying GAP readout')
    weights_conv = get_filter(
                        in_shape=X.shape,
                        out_channels=n_out,
                        ker_shape=ker_shape,
                        filter_init=W_init,
                        filter_reg=W_reg,
                        name='%s_1dconv'%(name),
                        )
    W_conv = weights_conv[0]
    if len(weights_conv)>1:
        B_conv = weights_conv[1]
    X = apply_conv(X, W_conv,
                    name='GAP_1dConv')
    X = apply_bias(X,B_conv,
                    name='GAP_in')

    n,h,w,n_in = X.shape.as_list()
    X = tf.nn.avg_pool(X,ksize=[1,h,w,1],
                        padding='VALID',
                        strides=[1,1,1,1])
    X = tf.squeeze(X, (1,2))
    return X

def frobenius_innerprod(A, B, n_ker):
    ## Function to implement Frobenius
    ## inner product on the matrices A
    ## and B. F = A' * B;
    ndims = len(A.shape)
    transpose_order = [ndims] + range(ndims-1)
    A_t = tf.transpose(A, transpose_order)
    A_r = tf.reshape(A_t,(n_ker, -1))
    B_t = tf.transpose(B, (4,0,1,2,3))
    B_r = tf.reshape(B_t, (n_ker,-1))
    # Normalizing to avoid gradient explosion
    A_norm = tf.nn.l2_normalize(A_r,axis=1)
    B_norm = tf.nn.l2_normalize(B_r,axis=1)
    B_norm = tf.transpose(B_norm,(1,0))
    F = tf.matmul(A_norm, B_norm)
    return F

def save_model_params(sess, expt_path, all_weights=False, w_to_save=[], iter=None):
    """Function to save model parameters
    :param sess: Session instance where model is built
    :param all_weights: Boolean set to save all weights
    :param w_to_save: List of names of weights to save
    """
    import datetime
    if all_weights:
        all_vars = [(v.name, v) for v in tf.trainable_variables()]
        all_vars = [(v[0].replace('/','_')\
                         .replace(':0',''),
                         sess.run(v[1])) for v in all_vars]
        date_time = ret_datetime()
        curr_run_path = os.path.join(expt_path,
                                    '%s-%s'%(date_time,iter))
        mkdir(curr_run_path)
        for (var,weight) in all_vars:
            curr_file = os.path.join(curr_run_path,
                                    '%s.npy'%(var))
            np.save(curr_file, weight)
    else:
        all_vars = [(v, sess.run(v)) for v in w_to_save]
        date_time = str(datetime.now())
        curr_run_path = os.path.join(expt_path,
                                    date_time)
        mkdir(curr_run_path)
        for (var,weight) in all_vars:
            curr_file = os.path.join(curr_run_path,
                                    '%s.npy'%(var))
            np.save(curr_file, weight)

def load_weights(sess, all_weights=True, weight_dr='.', slim=False):
    """Function to load saved weights from weight_dr
    :param sess: Session instance where model is built
    :param all_weights: Boolean set to load all weights
    :param weight_dr: Directory where all weights are stored
                    as numpy arrays."""
    print('Loading weights from %s'%(weight_dr))
    w_files = glob.glob('%s/*.npy'%(weight_dr))
    weights = {f:np.load(f) for f in w_files}
    weights = {k.split('/')[-1][:-4]:v
                    for k,v in weights.iteritems()}
    model2fn = {v.name:v.name.replace('/','_')[:-2]
                    for v in tf.trainable_variables()}
    all_w = {v.name:v for v in tf.trainable_variables()}
    for model_name, file_name in model2fn.iteritems():
        w = weights[file_name]
        sess.run(all_w[model_name].assign(w))
    print('Loaded weights from %s'%(weight_dr))
