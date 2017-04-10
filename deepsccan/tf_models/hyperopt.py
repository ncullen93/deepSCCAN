"""
Hyperparameter Optimization routines for CCA models
"""


import sys
import numpy as np
import scipy.stats 
import matplotlib.pyplot as plt

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from sparse_cca import SparseCCA
from deep_cca import DeepCCA
from conv_cca import ConvCCA


class HyperOptimizer(object):

    def __init__(self, model, hyper_space):
        self.model = model
        self.hyper_space = hyper_space

    def fit(self, x, y, x_val, y_val):
        pass

def hyperInit(x, y, hyper_space, nvecs=1, x_val=None, y_val=None, 
    max_evals=100, model='scca',verbose=0):

    def hyper_fn(params):
        model = SparseCCA(init_hyper=params, verbose=1)
        (corrs,val_corrs),_,_,_ = model.fit(x=x, y=y, x_val=x_val, y_val=y_val)
        loss = (nvecs - np.sum(val_corrs))**2 # minimize this
        print('Val Corrs:' ,  val_corrs)
        print('Loss: ' , loss)
        return {'loss': loss, 'status': STATUS_OK}

    hyper_params = {}
    for key, val in hyper_space.items():
        if type(val) is list or type(val) is tuple:
            if key == "algo":
                hyper_params['algo'] = hp.choice(key, val)
            elif "sparsity" in key or "smoothness" in key:
                hyper_params[key] = 10**(-hp.uniform(key,val[0],val[1]))
            else:
                hyper_params[key] = hp.uniform(key, val[0],val[1])
        else:
            hyper_params[key] = val
    trials = Trials()
    best = fmin(hyper_fn, hyper_params, algo=tpe.suggest, 
        max_evals=max_evals, trials=trials)
    return trials, best