"""
Script for running a Sparse CCA example on MNIST
"""
import os
os.chdir('c:/users/ncullen/desktop/projects/deepsccan/deepsccan')
from sparse_cca import SparseCCA

import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
from keras.datasets import mnist


### LOAD DATA AND PROCESS IT ###
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = (xtrain - np.mean(xtrain)) / np.std(xtrain)
x_array = xtrain[:,:,:14]
y_array = xtrain[:,:,14:]


####### BUILD MODEL #######
tf.reset_default_graph()
## Hyper-params
NVECS = 5
LEARN_RATE = 1e-4
NB_EPOCHS = 500
X_SPARSE, Y_SPARSE = 1e-6, 1e-6
X_SMOOTH, Y_SMOOTH = 1e-6, 1e-6

## functions
def l1l2_regularizer(l1_penalty, l2_penalty):
    def l1l2(weights):
        l1_score = tf.multiply(tf.reduce_sum(tf.abs(weights)), 
                        tf.convert_to_tensor(l1_penalty, dtype=tf.float32))
        l2_score = tf.multiply(tf.nn.l2_loss(weights),
                               tf.convert_to_tensor(l2_penalty, dtype=tf.float32))
        return tf.add(l1_score, l2_score)
    return l1l2

def dense_layer(x, units, activation, name, sparsity=0., smoothness=0.):
    if sparsity > 0. or smoothness > 0.:
        regularizer = l1l2_regularizer(l1_penalty=sparsity, l2_penalty=smoothness)
    else:
        regularizer = None
    y = tf.layers.dense(x, units=units, activation=activation,
            kernel_regularizer=regularizer, use_bias=False, name=name)
    # create clip norm function
    with tf.variable_scope(name, reuse=True):
        weights = tf.get_variable('kernel')
        clipped = tf.clip_by_norm(weights, clip_norm=1.0, axes=0)
        #l1_clipped = tf.clip_by_norm(l2_clipped, )
        clip_weights = tf.assign(weights, clipped, name='maxnorm')
        tf.add_to_collection('maxnorm', clip_weights)
    return y

def process_inputs(x, y):
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    return x, y

def tf_pearson_correlation(x_proj, y_proj):
    """
    Calculate pearson R correlation matrix between
    two matrices of shape (samples, nvecs) and returns
    the *nvecs* correlation coefficients btwn the corresponding
    i'th columns in each of the two matrices
    """
    mx = tf.reduce_mean(x_proj, axis=0)
    my = tf.reduce_mean(y_proj, axis=0)
    xm = x_proj - mx
    ym = y_proj - my
    r_num = tf.matmul(tf.transpose(xm), ym)
    r_den = tf.sqrt(tf.reduce_sum(tf.square(xm), axis=0) * tf.reduce_sum(tf.square(ym), axis=0))
    r_mat = tf.divide(r_num, r_den)
    r_vals = tf.diag_part(r_mat)
    return r_vals


def clip_by_l1_norm(t, clip_norm, axes=None, name=None):
    t = tf.convert_to_tensor(t, name='t')
    l1norm_inv = tf.reciprocal(tf.reduce_sum(tf.abs(t), axes, keep_dims=True))
    intermediate = t * clip_norm
    _ = t.shape.merge_with(intermediate.shape)
    tclip = tf.identity(intermediate * tf.minimum(l1norm_inv, 
                    tf.constant(1.0, dtype=t.dtype)) / clip_norm, name=name)
    return tclip

## process inputs
x_array, y_array = process_inputs(x_array, y_array)

## placeholders
x_place = tf.placeholder(tf.float32, shape=(None, x_array.shape[-1]), name='x_place')
y_place = tf.placeholder(tf.float32, shape=(None, y_array.shape[-1]), name='y_place')

## components
x_proj = dense_layer(x_place, units=NVECS, name='x_proj', 
                     sparsity=X_SPARSE, smoothness=X_SMOOTH)
y_proj = dense_layer(y_place, units=NVECS, name='y_proj',
                     sparsity=Y_SPARSE, smoothness=Y_SMOOTH)

## cca loss
# covariance matrix
covar_mat = tf.matmul(tf.transpose(x_proj), y_proj)
# sum of diagonal
diag_sum = tf.reduce_sum(tf.abs(tf.diag_part(covar_mat)))
# reverse sign for minimization
cca_loss = tf.multiply(-1., diag_sum)

## regularization losses
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_loss = tf.add_n(reg_losses)

## total loss
total_loss = tf.add(cca_loss, reg_loss)

## create optimizer and train op
optimizer = tf.train.AdamOptimizer(learning_rate=LEARN_RATE)
train_op = optimizer.minimize(total_loss)

## create eval op
eval_op = tf_pearson_correlation(x_proj, y_proj)

maxnorm_ops = tf.get_collection("maxnorm")

