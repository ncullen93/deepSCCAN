"""
Script for running a Sparse CCA example on MNIST
"""
import os
os.chdir('c:/users/ncullen/desktop/projects/deepsccan/deepsccan')
from sparse_cca import SparseCCA

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
from keras.datasets import mnist

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = xtrain - 255.
xx = xtrain[:20000,:,:14]
yy = xtrain[:20000,:,14:]

# Hyper-params
OPTIMIZER = 'adam'
LEARN_RATE = 1e-3
BATCH_SIZE = 32
NB_EPOCH = 1000
NVECS = 5

cca_model = SparseCCA(nvecs=NVECS, activation='linear')
#cca_model.fit(xx, yy, nb_epoch=5, batch_size=16)

x_array, y_array = cca_model.process_inputs(xx, yy)

# create placeholders
x_place = tf.placeholder(tf.float32, [1, None, x_array.shape[-1]])
y_place = tf.placeholder(tf.float32, [1, None, y_array.shape[-1]])

# create weight variables
x_proj, y_proj = cca_model._inference(x_place, y_place)
loss = cca_model._loss(x_proj, y_proj)
train_op = cca_model._train_op(loss, OPTIMIZER, LEARN_RATE)
corr_vals, eval_update_op = cca_model._eval_op(x_proj, y_proj)

# ops to clip weights so they have unit norm
maxnorm_ops = tf.get_collection("maxnorm")

nb_batches = int(np.ceil(x_array.shape[1] / float(BATCH_SIZE)))
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
cca_model.sess.run(init_op)

with tf.variable_scope('x_proj', reuse=True):
    x_weights = tf.get_variable('kernel')
with tf.variable_scope('y_proj', reuse=True):
    y_weights = tf.get_variable('kernel')

for epoch in range(NB_EPOCH):
    x_batches = np.array_split(x_array, nb_batches, axis=1)
    y_batches = np.array_split(y_array, nb_batches, axis=1)
    for x_batch, y_batch in zip(x_batches, y_batches):
        _loss, _, _ = cca_model.sess.run([loss, train_op, eval_update_op],
                             feed_dict={x_place: x_batch,
                                        y_place: y_batch})
        # clip max norm of weights
        cca_model.sess.run(maxnorm_ops)
        print('Loss: %.02f' % (_loss))
        c_vals = [cca_model.sess.run(c) for c in corr_vals]
        print('Avg Comp. Corr: %.02f' % np.mean([c_vals]))

xw = cca_model.sess.run(x_weights)
x_proj = np.dot(x_array, xw)[0]
yw = cca_model.sess.run(y_weights)
y_proj = np.dot(y_array, yw)[0]
for i in range(NVECS):
    plt.imshow(xw[:,i].reshape(28,28))
    plt.show()



#### CORRELATION EXAMPLE
from tensorflow.contrib.metrics import streaming_pearson_correlation

x = np.random.randn(10, 5)
y = np.random.randn(10, 5)
xx = tf.convert_to_tensor(x, dtype=tf.float32)
yy = tf.convert_to_tensor(y, dtype=tf.float32)
np_corr = [scipy.stats.pearsonr(x[:,i], y[:,i])[0] for i in range(x.shape[1])]

corrs = []
for i in range(x.shape[1]):
    corr, update_op = streaming_pearson_correlation(xx[:,i], yy[:,i])
    corrs.append((corr, update_op))

init_op = tf.group(tf.global_variables_initializer(), 
            tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init_op)
    for a, b in corrs:
        sess.run(b)
        print(sess.run(a))


