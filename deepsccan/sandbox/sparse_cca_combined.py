"""
from sklearn.datasets import fetch_olivetti_faces
data = fetch_olivetti_faces()
images = data.images
x_array = images[:,:,:32].reshape(images.shape[0], -1)
y_array = images[:,:,32:].reshape(images.shape[0], -1)

"""
import tensorflow as tf
import numpy as np
np.set_printoptions(suppress=True)

import matplotlib.pyplot as plt

from keras.datasets import mnist
(xtrain,ytrain),(xtest,ytest) = mnist.load_data()
xtrain = xtrain / 255.

x = xtrain[:5000]
y = np.zeros((5000,1))
for i in range(5000):
    if ytrain[i] in [2]:
        y[i] = 1.
    else:
        y[i] = 0.

x_array = x.reshape(x.shape[0], -1)
y_array = y

tf.reset_default_graph()
L1_PENALTY = 0.5
LEARN_RATE = 3e-4
NVECS = 8
NB_EPOCH = 500
BATCH_SIZE = 500

nb_batches = int(np.ceil(x_array.shape[0]/BATCH_SIZE))
xy_array = []
for batch_idx in range(nb_batches):
    x_batch = x_array[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
    y_batch = y_array[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
    xy_dot = np.dot(x_batch.T, y_batch)
    xy_array.append(xy_dot)
xy_array = np.asarray(xy_array)

#x1 = tf.placeholder(tf.float32, shape=(None, x_array.shape[-1]))
#x2 = tf.placeholder(tf.float32, shape=(None, y_array.shape[-1]))

x1tx2 = tf.placeholder(tf.float32, shape=(x_array.shape[-1], y_array.shape[-1]))

u = tf.get_variable('u', shape=(x_array.shape[-1], 1))
v = tf.get_variable('v', shape=(y_array.shape[-1], 1))

#x1tx2 = tf.matmul(tf.transpose(x1), x2)
ux1tx2 = tf.matmul(tf.transpose(u), x1tx2)
ux1tx2v = tf.matmul(ux1tx2, v)
cca_loss = -ux1tx2v

## clipping
clipped_u = tf.clip_by_norm(u, clip_norm=1.0, axes=0)
clip_u = tf.assign(u, clipped_u, name='ortho')
tf.add_to_collection('normalize_ops', clip_u)
clipped_v = tf.clip_by_norm(v, clip_norm=1.0, axes=0)
clip_v = tf.assign(v, clipped_v, name='ortho')
tf.add_to_collection('normalize_ops', clip_v)

## l1 penalty
l1_u = tf.reduce_sum(tf.abs(u))
l1_v = tf.reduce_sum(tf.abs(v))

## total loss
total_loss = cca_loss + l1_u * L1_PENALTY + l1_v * L1_PENALTY

## optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARN_RATE)
train_op = optimizer.minimize(total_loss)

init_op = tf.global_variables_initializer()
normalize_ops = tf.get_collection('normalize_ops')

xw = np.zeros((x_array.shape[-1], NVECS))
yw = np.zeros((y_array.shape[-1], NVECS))
with tf.Session() as sess:
    for nvec in range(NVECS):
        print('Component ', nvec)
        # re-initialize everything
        sess.run(init_op)
        #plt.imshow(xy_array[0])
        #plt.show()

        for epoch in range(NB_EPOCH):
            c_losses = []
            for batch_idx in range(xy_array.shape[0]):
                x1tx2_batch = xy_array[batch_idx]

                t_loss, c_loss, _ = sess.run([total_loss, cca_loss, train_op],
                                   feed_dict={x1tx2: x1tx2_batch})

                # clip weights to have unit l2 norms
                sess.run(normalize_ops)
                c_losses.append(c_loss)

            #print('Epoch Loss: %.02f' % np.mean(c_losses))

        # get components
        with tf.variable_scope('', reuse=True):
            uu = sess.run(u)
            vv = sess.run(v)
            xw[:,nvec] = np.squeeze(uu)
            yw[:,nvec] = np.squeeze(vv)

        # deflate the inputs
        for i in range(xy_array.shape[0]):
            xy_array[i] = xy_array[i] - uu.T.dot(xy_array[i]).dot(vv) * np.dot(uu, vv.T)

#plt.imshow(xy_array[0])
#plt.show()
for i in range(xw.shape[-1]):
    plt.imshow(xw[:,i].reshape(28,28))
    plt.show()
    #plt.imshow(yw[:,i].reshape(64,32))
    #plt.show()


xproj = np.dot(x_array, xw)
yproj = np.dot(y_array, yw)
covar = np.dot(xproj.T, yproj)
print(np.round(covar,2))