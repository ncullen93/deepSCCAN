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

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = (xtrain - np.mean(xtrain)) / np.std(xtrain)
xx = xtrain[:30000,:,:14]
yy = xtrain[:30000,:,14:]

# Hyper-params
NVECS = 5

cca_model = SparseCCA(nvecs=NVECS, activation='linear',
                      sparsity=(1e-5,1e-5))
cca_model.fit(xx, yy, nb_epoch=1000, batch_size=64, learn_rate=1e-3, verbose=False)

#xw, yw = cca_model.get_weights()
with tf.variable_scope('', reuse=True):
    xw = cca_model.sess.run(tf.get_variable('x_proj/kernel'))
    yw = cca_model.sess.run(tf.get_variable('y_proj/kernel'))

#for i in range(xw.shape[-1]):
#    plt.imshow(xw[:,i].reshape(28,14))
#    plt.show()

xproj = np.dot(xx.reshape(xx.shape[0],-1), xw)
yproj = np.dot(yy.reshape(yy.shape[0],-1), yw)
cvals = np.zeros(NVECS)
for i in range(NVECS):
    cvals[i] = scipy.stats.pearsonr(xproj[:,i],yproj[:,i])[0]
print(cvals)
print('\n')
print(np.round(np.corrcoef(xw),2))

PLOT = False
if PLOT:
    for i in range(xw.shape[-1]):
        plt.imshow(xw[:,i].reshape(28,14))
        plt.show()
