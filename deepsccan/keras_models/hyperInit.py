"""
Hyperparameter optimization to initialize the
parameters of a sparse or deep CCA model
"""

# 3RD PARTY IMPORTS
import sys
import numpy as np
import scipy.stats 
import matplotlib.pyplot as plt

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from sparseDecom2   import sparseDecom2
from kernelDecom2   import kernelDecom2
from deepDecom2     import deepDecom2
from convDecom2     import convDecom2


def hyperInit(inmats, hyper_space, nvecs=1, valmats=None, 
    max_evals=100, model='scca',verbose=0):

    def hyper_fn(params):
        (corrs,val_corrs),_,_,_ = sparseDecom2(inmats=inmats,valmats=valmats,
            nvecs=nvecs, init_hyper=params, verbose=1)
        loss = (nvecs - np.sum(val_corrs))**2 # minimize this
        print 'Val Corrs:' ,  val_corrs
        print 'Loss: ' , loss
        return {'loss': loss, 'status': STATUS_OK}

    inmats  = inmats
    valmats = valmats
    nvecs   = nvecs

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









