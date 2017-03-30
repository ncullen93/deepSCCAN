"""
Sparse CCA code
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
from keras.layers       import TimeDistributed, Dense, Merge, Lambda, Reshape
from keras.callbacks    import ModelCheckpoint, EarlyStopping
from keras.constraints  import UnitNorm
from keras.regularizers import l1l2
from keras.layers.normalization import BatchNormalization


def deepDecom2(inmats, nvecs, hidden_layers,
        sparsity=(1e-3,1e-3), smoothness=(1e-4,1e-4), 
        its=500, algo='nadam', verbose=0, nonneg=False):
    """
    Run Sparse CCA.

    Arguments
    ---------
    inmats      : a tuple or list
        (dataset1, dataset2) -> train datasets of shape (subjects,features)

    nvecs       : an integer
        The number of final components to learn

    hidden_layers   : a list of list of tuples
        number of hidden components for X and Y -> ex [[(50,'sigmoid')],[(10,'sigmoid')]]

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

    # data must be in three dimensions
    if X.ndim == 2:
        X = X.reshape(1,X.shape[0], X.shape[1])
    if Y.ndim == 2:
        Y = Y.reshape(1, Y.shape[0], Y.shape[1])

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
    params['hidden_Y']      = hidden_layers[1]

    # FIT MODEL 
    corrs, xp, yp = fit_dcca_model(current_inputs=current_inputs, params=params)

    return corrs, xp, yp


def fit_dcca_model(current_inputs, params):
    """
    Compile and Fit keras model. Return corrs, comps, projs
    """
    model_loss  = cca_loss
    outputs     = np.array([[[0.]]])

    # build model
    model = build_dcca_model(params)

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

def build_dcca_model(params):
    """
    Build and return an UNCOMPILED keras model
    """
    ####################
    # X projection model
    ####################
    modelX = Sequential()
    # add hidden layers
    has_hidden=False
    for x_idx, (neurons, activity) in enumerate(params['hidden_X']):
        has_hidden=True
        # need to include input_shape in first layer
        if x_idx == 0:
            modelX.add(TimeDistributed(Dense(neurons, bias=False,
                W_regularizer=l1l2(l1=params['sparsity_X'],l2=params['smooth_X']),
                activation=activity),
                input_shape=(params['shape_X'][1],params['shape_X'][2])))
        else:
            modelX.add(TimeDistributed(Dense(neurons, bias=False,
                W_regularizer=l1l2(l1=params['sparsity_X'],l2=params['smooth_X']),
                activation=activity)))
    if has_hidden == False:
        modelX.add(TimeDistributed(Dense(params['nvecs'], bias=False,
            activation=params['final_act'],
            W_regularizer=l1l2(l1=params['sparsity_X'],l2=params['smooth_X'])),
            input_shape=(params['shape_X'][1],params['shape_X'][2])))
    else:
        modelX.add(TimeDistributed(Dense(params['nvecs'], bias=False,
            activation=params['final_act'],
            W_regularizer=l1l2(l1=params['sparsity_X'],l2=params['smooth_X']))))        

    ####################
    # Y projection model
    ####################
    modelY = Sequential()
    # add hidden layers
    has_hidden=False
    for y_idx, (neurons, activity) in enumerate(params['hidden_Y']):
        has_hidden=True
        # need to include input_shape in first layer
        if y_idx == 0:
            modelY.add(TimeDistributed(Dense(neurons, bias=False,
                W_regularizer=l1l2(l1=params['sparsity_Y'],l2=params['smooth_Y']),
                activation=activity),
                input_shape=(params['shape_Y'][1],params['shape_Y'][2])))
        else:
            modelY.add(TimeDistributed(Dense(neurons, bias=False,
                W_regularizer=l1l2(l1=params['sparsity_Y'],l2=params['smooth_Y']),
                activation=activity)))
    # add final layer
    if has_hidden == False:
        modelY.add(TimeDistributed(Dense(params['nvecs'], bias=False,
            activation=params['final_act'],
            W_regularizer=l1l2(l1=params['sparsity_Y'],l2=params['smooth_Y'])),
            input_shape=(params['shape_Y'][1],params['shape_Y'][2])))
    else:
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





