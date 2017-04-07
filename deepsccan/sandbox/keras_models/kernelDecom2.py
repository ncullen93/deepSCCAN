"""
Nonlinear Sparse CCA code
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


def kernelDecom2(inmats, nvecs, mycoption=0,
            activation='sigmoid', sparsity=(1e-4,1e-4), 
            smoothness=(1e-4,1e-4), its=500, 
            myconstraint=0, algo='nadam', verbose=0, nonneg=True):
    """
    Run Nonlinear Sparse CCA.

    Arguments
    ---------
    inmats      : a tuple or list
        (dataset1, dataset2) -> train datasets of shape (subjects,features)

    nvecs       : an integer
        The number of components to learn

    mycoption   : an integer
        0 = batch orthogonality method
        1 = deflation method

    sparsity    : a tuple or list
        (c1,c2) -> penalty on L1 norm

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
    params['nvecs']         = nvecs
    params['mycoption']     = mycoption
    params['myconstraint']  = myconstraint
    params['activation']    = activation



    if mycoption not in [0,1]:
        print 'mycoption argument invalid.. Using mycoption=0'
        params['mycoption'] = 0

    if myconstraint not in [0,1]:
        print 'myconstraint argument invalid.. Using mycoption=0'
        params['myconstraint'] = 0

    # FIT MODEL 
    corrs, (u_comp, v_comp), (x_proj, y_proj) = fit_scca_model(current_inputs=current_inputs, 
                                                                params=params)

    return corrs, (u_comp, v_comp), (x_proj, y_proj)


def fit_scca_model(current_inputs, params):
    """
    Compile and Fit keras model. Return corrs, comps, projs
    """

    orig_inputs = [current_inputs[0][0,:,:].copy(), current_inputs[1][0,:,:].copy()] # save for corrs/deflate

    # batch method
    if params['mycoption'] == 0:
        model_loss  = cca_loss
        outputs     = np.array([[[0.]]])

        # build model
        model = build_scca_model(params)

        # compile model
        model.compile(optimizer=params['algo'], loss=model_loss)

        # fit model
        model.fit(current_inputs, outputs, nb_epoch=params['its'],
                verbose=params['keras_verbose'])

        # extract weights from trained keras model
        u_comp  = model.layers[0].layers[0].layers[0].get_weights()[0]
        v_comp  = model.layers[0].layers[1].layers[0].get_weights()[0]

        # make projections
        x_proj  = np.dot(orig_inputs[0], u_comp)
        y_proj  = np.dot(orig_inputs[1], v_comp)

        # calculate correlations
        corrs   = np.array([scipy.stats.pearsonr(x_proj[:,i],y_proj[:,i])[0] for i in range(params['nvecs'])])

    # deflate method
    elif params['mycoption'] == 1:
        
        model_loss  = cca_loss
        outputs     = np.array([[[0.]]])

        u_comp = []
        v_comp = []
        for i in range(params['nvecs']):

            # build model
            model = build_scca_model(params)

            # compile model
            model.compile(optimizer=params['algo'], loss=model_loss)

            # fit model
            model.fit(current_inputs, outputs, nb_epoch=params['its'],
                    verbose=params['keras_verbose'])

            # extract weights and add to component list
            _u_comp     = model.layers[0].layers[0].layers[0].get_weights()[0]
            _v_comp     = model.layers[0].layers[1].layers[0].get_weights()[0]
            u_comp.append(np.squeeze(_u_comp))
            v_comp.append(np.squeeze(_v_comp))
            
            # deflate inputs
            current_inputs = deflate_inputs(current_inputs, orig_inputs, _u_comp, _v_comp)


        # make projections
        u_comp  = np.array(u_comp)
        v_comp  = np.array(v_comp)
        x_proj  = np.dot(orig_inputs[0], u_comp.T)
        y_proj  = np.dot(orig_inputs[1], v_comp.T)

        # calculate correlations
        corrs   = np.array([scipy.stats.pearsonr(x_proj[:,i],y_proj[:,i])[0] for i in range(params['nvecs'])])


    return corrs, (u_comp, v_comp), (x_proj, y_proj)

def build_scca_model(params):
    """
    Build and return an UNCOMPILED keras model
    """
    if params['mycoption'] == 0:
        dense_vecs  = params['nvecs'] # batch method - learn all together
    elif params['mycoption'] == 1:
        dense_vecs  = 1 # deflate method - learn one at a time

    ####################
    # X projection model
    ####################
    modelX = Sequential()
    modelX.add(TimeDistributed(Dense(dense_vecs, bias=False, 
        #W_constraint=UnitNorm(),
        activation=params['activation'],
        W_regularizer=l1l2(l1=params['sparsity_X'],l2=params['smooth_Y'])),
        input_shape=(params['shape_X'][1],params['shape_X'][2])))

    ####################
    # Y projection model
    ####################
    modelY = Sequential()
    modelY.add(TimeDistributed(Dense(dense_vecs, bias=False,
        #W_constraint=UnitNorm(),
        activation=params['activation'],
        W_regularizer=l1l2(l1=params['sparsity_X'],l2=params['smooth_X'])),
        input_shape=(params['shape_Y'][1],params['shape_Y'][2])))

    ##############
    # merged model
    ##############
    merged_model = Sequential()
    merged_model.add(Merge([modelX,modelY], mode='dot', dot_axes=1))
    if params['mycoption'] == 0:
        merged_model.add(Lambda(lambda x: batch_cca_loss(x,0.), output_shape=(1,1)))

    return merged_model


def deflate_inputs(current_inputs, original_inputs, u, v):
    Xp      = current_inputs[0][0,:,:]
    Yp      = current_inputs[1][0,:,:]
    X_orig  = original_inputs[0]
    Y_orig  = original_inputs[1]
    #u      = u / (np.sqrt(np.sum(u**2)+1e-7))
    #v      = v / (np.sqrt(np.sum(v**2)+1e-7))

    qx = np.dot(Xp.T,np.dot(X_orig,u))
    qx = qx / (np.sqrt(np.sum(qx**2)+1e-7))
    #qx = qx.astype('float16')
    Xp = Xp - np.dot(Xp,qx).dot(qx.T)
    X  = Xp.reshape(1,Xp.shape[0],Xp.shape[1])

    qy = np.dot(Yp.T,np.dot(Y_orig,v))
    qy = qy / (np.sqrt(np.sum(qy**2)+1e-7))
    #qy = qy.astype('float16')
    Yp = Yp - np.dot(Yp,qy).dot(qy.T)
    Y  = Yp.reshape(1,Yp.shape[0],Yp.shape[1])

    new_current_inputs = [X,Y]
    return new_current_inputs