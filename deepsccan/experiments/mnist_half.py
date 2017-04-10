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
x_train = train_imgs[:-6000,:,:14]
y_train = train_imgs[:-6000,:,14:]
x_val = train_imgs[-6000:,:,:14]
y_val = train_imgs[-6000:,:,:14]
x_test = test_imgs[:,:,:14]
y_test = test_imgs[:,:,14:]

# -------------------------------------------------------------------
## build, fit, and evaluate the Sparse CCA model

# make model architecture
cca_model = SparseCCA(nvecs=50, activation='linear', sparsity=(1e-4, 1e-4), deflation=False)

# fit model on data
cca_model.fit(x=x_train, y=y_train, nb_epoch=200, batch_size=256, learn_rate=5e-4)

# evaluate model on validation data
corr_vals = cca_model.evaluate(x=x_test, y=y_test)
print('Mean Corr Vals: ', np.mean(corr_vals))
