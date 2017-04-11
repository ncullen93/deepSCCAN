"""
Experiment to learn correlated representations of the
left and right halves of MNSIT digits w/ Sparse CCA
"""
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# -------------------------------------------------------------------
## Load and Process Data

from keras.datasets import mnist
(train_imgs, train_lbls), (test_imgs, test_lbls) = mnist.load_data()

# standardization to zero mean and unit variance
train_imgs = (train_imgs - np.mean(train_imgs)) / np.std(train_imgs)
test_imgs = (test_imgs - np.mean(test_imgs)) / np.std(test_imgs)

# cut images in half -> 54k train, 6k val, 10k test
x_train = train_imgs[:,:,:14]
y_train = train_imgs[:,:,14:]
x_test = test_imgs[:,:,:14]
y_test = test_imgs[:,:,14:]

# -------------------------------------------------------------------
## build, fit, and evaluate the Sparse CCA model

# make model architecture
cca_model = SparseCCA(nvecs=50, nb_epoch=10)

# make hyper-param grid
param_grid = {
    'learn_rate': [1e-3, 1e-4, 1e-5],
    #'nb_epoch': [10],
    'sparsity': [1, 0.1, 1e-2, 1e-4],
    'deflation': [False, True]
}

# split train data into train and validation sets
test_fold = [-1 if i<(0.9*x_train.shape[0]) else 0 for i in range(x_train.shape[0])]
cv = PredefinedSplit(test_fold=test_fold)

# instantiate grid search model
grid_model = GridSearchCV(cca_model, param_grid=param_grid, cv=cv,
                n_jobs=1, verbose=0)

grid_model.fit(x_train, y_train)

