"""
Convolutional CCA model
"""

# SYSTEM IMPORTS
import os
import shutil

# 3RD PARTY IMPORTS
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# CUSTOM KERAS IMPORTS
import keras_custom
from keras_custom import cca_loss, batch_cca_loss, CCA_Constraint, PickMax

# KERAS IMPORTS
import keras
import keras.backend as K
from keras.models       import Sequential, Model
from keras.layers       import TimeDistributed, Dense, Merge, Lambda, Convolution2D
from keras.layers       import Flatten, MaxPooling2D
from keras.callbacks    import ModelCheckpoint, EarlyStopping
from keras.constraints  import UnitNorm
from keras.regularizers import l1l2
from keras.layers.normalization import BatchNormalization


def convDecom2(inmats, nvecs, conv_layers, hidden_layers,
        sparsity=(1e-3,1e-3), smoothness=(1e-4,1e-4), 
        its=500, algo='nadam', verbose=0, nonneg=False):
    """
    Run Convolutional CCA

    Arguments
    ---------
    inmats      : a tuple or list
        (dataset1, dataset2) -> train datasets of shape (subjects,features)

    nvecs       : an integer
        The number of final components to learn

    conv_layers : a list of list of tuples
        number of hidden components for X and Y.
        Each tuple should be (n_kernels, kernel_size_x, kernel_size_y, activation, max_pool)
            n_kernels, kernel_size_x, kernel_size_y = Integers
            activation = string
            max_pool = Boolean (whether to include 2x2 max pooling afterwards)

    sparsity    : a tuple or list
        (c1,c2) -> sparsity values = Max L1 Norm allowed

    smoothness  : a tuple or list
        (s1,s2) -> penalty on L2 norm

    its         : an integer
        number of iterations to run the optimizer

    algo        : a string or Keras.optimizer object
        which gradient descent algorithm to use

    verbose     : an integer
        -1  = print nothing
        0   = print correlations after each run
        1   = print full Keras training status

    nonneg      : boolean
        whether to make weights of components non-negative

    Returns
    -------
    corrs   : 1D numpy array
        correlations for each component on training set
    
    (u_comp, v_comp)            : tuple of 2D numpy arrays
        the components learned on X and Y, respectively - shape = (nvecs, features)

    (x_proj, y_proj)            : tuple of 2D numpy arrays
        the projections of components on X and Y, respectively - shape = (subjects, nvecs)

    
    """
    X = inmats[0]
    Y = inmats[1]

    # reshape data for keras
    X = X.reshape(1,X.shape[0], 1, X.shape[1], X.shape[2])
    Y = Y.reshape(1,Y.shape[0], 1, Y.shape[1], Y.shape[2])

    current_inputs  = [X,Y]

    # set params dictionary
    params = {}
    params['shape_X']       = X.shape
    params['shape_Y']       = Y.shape
    params['sparsity_X']    = sparsity[0]
    params['sparsity_Y']    = sparsity[1]
    params['smooth_X']      = smoothness[0]
    params['smooth_Y']      = smoothness[1]
    params['algo']          = algo
    params['its']           = its
    params['keras_verbose'] = max(0,verbose)
    params['nonneg']        = nonneg
    params['nvecs']         = nvecs[0]
    params['final_act']     = nvecs[1]
    params['hidden_X']      = hidden_layers[0]
    params['conv_X']        = conv_layers[0]
    params['hidden_Y']      = hidden_layers[1]
    params['conv_Y']        = conv_layers[1]

    # FIT MODEL 
    corrs, xp, yp = fit_convcca_model(current_inputs=current_inputs, params=params)

    return corrs, xp, yp


def fit_convcca_model(current_inputs, params):
    """
    Compile and Fit keras model. Return corrs, comps, projs
    """
    model_loss  = cca_loss
    outputs     = np.array([[[0.]]])

    # build model
    model = build_convcca_model(params)

    # compile model
    model.compile(optimizer=params['algo'], loss=model_loss)

    # fit model
    history = model.fit(current_inputs, outputs, nb_epoch=params['its'],
            verbose=params['keras_verbose'])

    # extract weights from trained keras model
    #u_comp     = model.layers[0].layers[0].layers[0].get_weights()[0]
    #v_comp     = model.layers[0].layers[1].layers[0].get_weights()[0]

    # make projections
    orig_x  = current_inputs[0]
    x_proj  = model.model.layers[0].predict(orig_x)[0,:,:]

    orig_y  = current_inputs[1]
    y_proj  = model.model.layers[1].predict(orig_y)[0,:,:]

    corrs = []
    for i in range(x_proj.shape[-1]):
        for j in range(y_proj.shape[-1]):
            if i==j:
                corrs.append(scipy.stats.pearsonr(x_proj[:,i],y_proj[:,j]))
    corrs = np.array(corrs)

    return corrs, x_proj, y_proj

def build_convcca_model(params):
    """
    Build and return an UNCOMPILED keras model
    """
    ####################
    # X projection model
    ####################
    modelX = Sequential()
    # add convolutional layers
    for x_idx, (nk,ks,act,mp) in enumerate(params['conv_X']):
        # need to include input_shape in first layer
        if x_idx == 0:
            modelX.add(TimeDistributed(Convolution2D(nk,ks,ks,
                W_regularizer=l1l2(l1=params['sparsity_X'],l2=params['smooth_X']),
                activation=act),
                input_shape=(params['shape_X'][1], params['shape_X'][2], 
                    params['shape_X'][3],params['shape_X'][4])))
            # add max pooling
            if mp == True:
                modelX.add(TimeDistributed(MaxPooling2D((2,2))))
        else:
            modelX.add(TimeDistributed(Convolution2D(nk,ks,ks,
                W_regularizer=l1l2(l1=params['sparsity_X'],l2=params['smooth_X']),
                activation=act)))
            # add max pool
            if mp == True:
                modelX.add(TimeDistributed(MaxPooling2D((2,2))))
    # flatten last conv layer
    modelX.add(TimeDistributed(Flatten()))
    # add fully connected layers
    for neurons, activity in params['hidden_X']:
        modelX.add(TimeDistributed(Dense(neurons, bias=False,
            W_regularizer=l1l2(l1=params['sparsity_Y'],l2=params['smooth_Y']),
            activation=activity)))
    # add final layer
    modelX.add(TimeDistributed(Dense(params['nvecs'], bias=False,
        activation=params['final_act'],
        W_regularizer=l1l2(l1=params['sparsity_X'],l2=params['smooth_X']))))

    ####################
    # Y projection model
    ####################
    modelY = Sequential()
    # add convolutional layers
    for y_idx, (nk,ks,act,mp) in enumerate(params['conv_Y']):
        # need to include input_shape in first layer
        if y_idx == 0:
            modelY.add(TimeDistributed(Convolution2D(nk,ks,ks,
                W_regularizer=l1l2(l1=params['sparsity_Y'],l2=params['smooth_Y']),
                activation=act),
                input_shape=(params['shape_Y'][1], params['shape_Y'][2],
                    params['shape_Y'][3], params['shape_Y'][4])))
            if mp == True:
                modelY.add(TimeDistributed(MaxPooling2D((2,2))))
        else:
            modelY.add(TimeDistributed(Convolution2D(nk,ks,ks,
                W_regularizer=l1l2(l1=params['sparsity_Y'],l2=params['smooth_Y']),
                activation=act)))
            if mp == True:
                modelY.add(TimeDistributed(MaxPooling2D((2,2))))
    # flatten last conv layer
    modelY.add(TimeDistributed(Flatten()))
    # add fully connected layers
    for neurons, activity in params['hidden_Y']:
        modelY.add(TimeDistributed(Dense(neurons, bias=False,
            W_regularizer=l1l2(l1=params['sparsity_Y'],l2=params['smooth_Y']),
            activation=activity)))
    # add final layer
    modelY.add(TimeDistributed(Dense(params['nvecs'], bias=False,
        activation=params['final_act'],
        W_regularizer=l1l2(l1=params['sparsity_Y'],l2=params['smooth_Y']))))

    ##############
    # merged model
    ##############
    merged_model = Sequential()
    merged_model.add(Merge([modelX,modelY], mode='dot', dot_axes=1))
    merged_model.add(Lambda(lambda x: batch_cca_loss(x,0.), output_shape=(1,1)))

    return merged_model





