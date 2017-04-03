"""
Simple example - learn one component
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def generate_simple(nsubjects=50, x_fts=200, y_fts=300, ncomp=1, visualize=False):
    v1 = np.vstack((np.ones((25,1)),-1*np.ones((25,1)),np.zeros((150,1))))
    v2 = np.vstack((np.zeros((250,1)),np.ones((25,1)),-1*np.ones((25,1))))
    u = np.random.normal(0, 1, (50,1))
    e1 = np.random.normal(0, 0.1**2, (200,1))
    e2 = np.random.normal(0, 0.1**2, (300,1))

    X = np.dot(v1+e1, u.T)
    Y = np.dot(v2+e2, u.T)

    if visualize:
        plt.scatter(np.arange(len(v1)), v1)
        plt.show()
        plt.scatter(np.arange(len(v2)), v2)
        plt.show()

        # correlations between componens
        #x_proj = np.dot(X.T, v1)
        #y_proj = np.dot(Y.T, v2)

        #for i in range(x_proj.shape[-1]):
        #    corr_val = scipy.stats.pearsonr(x_proj[:,i], y_proj[:,i])[0]
        #    print('Corr %i : %.02f' % (i, corr_val))

    return X, Y