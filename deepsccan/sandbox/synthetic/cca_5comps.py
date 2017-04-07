"""
Traditional (No Sparsity) CCA to learn FIVE ground truth components.

Here, we use the iterative approach to learning the components, instead
of learning them all at once.
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

import tensorflow as tf

np.random.seed(3636)


def generate_data(ncomps=2):
    v1 = np.zeros((200,2))
    v1[:25,0] = 1
    v1[25:50,0] = -1
    v1[100:120,1] = 1
    v1[140:160,1] = -1

    v2 = np.zeros((300,2))
    v2[250:275,0] = 1
    v2[275:,0] = -1
    v2[:20,1] = 1
    v2[30:60,1] = -1

    u = np.random.normal(0, 1, (500,2))
    e1 = np.random.normal(0, 0.01**2, (200,2))
    e2 = np.random.normal(0, 0.01**2, (300,2))

    X = np.dot(v1+e1, u.T)
    X = (X - np.mean(X)) / np.std(X)
    Y = np.dot(v2+e2, u.T)
    Y = (Y - np.mean(Y)) / np.std(Y)

    return X, Y, v1, v2, u

## BUILD MODEL ##
def _dense_layer(x, units, smoothness=1., name='dense'):
    #if sparsity > 0. or smoothness > 0.:
    #    regularizer = l1_regularizer(l1_penalty=sparsity)
    #else:
    regularizer = None
    
    y = tf.layers.dense(x, units=units, activation=None,
            kernel_regularizer=regularizer, use_bias=False, name=name)
    # create clip norm function
    with tf.variable_scope(name, reuse=True):
        weights = tf.get_variable('kernel')
        ss = tf.convert_to_tensor(smoothness, dtype=tf.float32)
        clipped = weights * tf.sqrt(ss) * tf.rsqrt(tf.reduce_sum(tf.square(weights)))
        #clipped = tf.clip_by_norm(weights, clip_norm=1.0, axes=0)
        clip_weights = tf.assign(weights, clipped, name='maxnorm')
        tf.add_to_collection('maxnorm', clip_weights)

    return y


def dense_layer(x, units, smoothness=1., name='dense'):
    #if sparsity > 0. or smoothness > 0.:
    #    regularizer = l1_regularizer(l1_penalty=sparsity)
    #else:
    regularizer = None
    
    y = tf.layers.dense(x, units=units, activation=None,
            kernel_regularizer=regularizer, use_bias=False, name=name)
    # create clip norm function
    with tf.variable_scope(name, reuse=True):
        weights = tf.get_variable('kernel')
        ysquared = tf.rsqrt(tf.reduce_sum(tf.square(y),axis=0))
        orthogonalized_weights = weights * ysquared
        ortho_weights = tf.assign(weights, orthogonalized_weights, name='ortho')
        tf.add_to_collection('ortho', ortho_weights)

    return y

def process_inputs(x, y):
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    return x, y 

def tf_pearson_correlation(x_proj, y_proj):
    """
    Calculate pearson R correlation matrix between
    two matrices of shape (samples, nvecs) and returns
    the *nvecs* correlation coefficients btwn the corresponding
    i'th columns in each of the two matrices
    """
    mx = tf.reduce_mean(x_proj, axis=0)
    my = tf.reduce_mean(y_proj, axis=0)
    xm = x_proj - mx
    ym = y_proj - my
    r_num = tf.matmul(tf.transpose(xm), ym)
    r_den = tf.sqrt(tf.reduce_sum(tf.square(xm), axis=0) * tf.reduce_sum(tf.square(ym), axis=0))
    r_mat = tf.divide(r_num, r_den)
    r_vals = tf.diag_part(r_mat)
    return r_vals

def matrix_deflation(X_curr, Y_curr, X_orig, Y_orig, u, v):
    """
    Deflate matrices and return new ones

    #def deflate_inputs(current_inputs, original_inputs, u, v):
    """
    Xp = X_curr
    Yp = Y_curr
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

    return X, Y

def matrix_deflation_2(X_old, Y_old, u, v):
    alpha = u.T.dot(X_old).dot(Y_old.T).dot(v)
    X_new = X_old - np.sqrt(alpha).dot(u)
    Y_new = Y_old - np.sqrt(alpha).dot(v)
    return X_new, Y_new

def matrix_deflation_3():
    pass

## Hyper-params
NVECS = 1
LEARN_RATE = 5e-4
NB_EPOCH = 1000
BATCH_SIZE = 500
X_SMOOTH = Y_SMOOTH = 1.

x_array, y_array, v1, v2, u = generate_data()
real_x_proj = np.dot(x_array.T, v1)
real_y_proj = np.dot(y_array.T, v2)

x_array, y_array = process_inputs(x_array.T, y_array.T)

tf.reset_default_graph()

x_place = tf.placeholder(tf.float32, shape=(None, x_array.shape[-1]), name='x_place')
y_place = tf.placeholder(tf.float32, shape=(None, y_array.shape[-1]), name='y_place')

x_proj = dense_layer(x_place, units=NVECS, smoothness=X_SMOOTH, name='x_proj')
y_proj = dense_layer(y_place, units=NVECS, smoothness=Y_SMOOTH, name='y_proj')

covar_mat = tf.matmul(tf.transpose(x_proj), y_proj)
cca_loss = tf.reduce_sum(tf.diag_part(tf.abs(covar_mat)))
upper_loss = tf.reduce_sum(tf.matrix_band_part(tf.abs(covar_mat), 0, -1))
lower_loss = tf.reduce_sum(tf.matrix_band_part(tf.abs(covar_mat), -1, 0))
total_loss = -3.*cca_loss + upper_loss + lower_loss

## create optimizer and train op
#optimizer = tf.train.MomentumOptimizer(learning_rate=LEARN_RATE, momentum=0.9)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARN_RATE)
train_op = optimizer.minimize(total_loss)

## create eval op
eval_op = tf_pearson_correlation(x_proj, y_proj)
## get weight clipping ops
#maxnorm_ops = tf.get_collection("maxnorm")
ortho_ops = tf.get_collection('ortho')

x_array_deflated = x_array.copy()
y_array_deflated = y_array.copy()

with tf.Session() as sess:
    nb_batches = int(np.ceil(x_array.shape[0]/BATCH_SIZE))

    for nvec in range(NVECS):
        sess.run(tf.global_variables_initializer())

        for epoch in range(NB_EPOCH):
            for batch_idx in range(nb_batches):
                x_batch = x_array[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
                y_batch = y_array[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]

                cl, loss, _ = sess.run([cca_loss, total_loss, train_op],
                                    feed_dict={x_place:x_batch,
                                               y_place:y_batch})
                #sess.run(maxnorm_ops)
                sess.run(ortho_ops, feed_dict={x_place:x_batch, y_place:y_batch})
                print('Loss %.02f, %.02f' % (cl, loss))

        # get components and projects
        with tf.variable_scope('', reuse=True):
            xw = sess.run(tf.get_variable('x_proj/kernel'))
            yw = sess.run(tf.get_variable('y_proj/kernel'))


#x, y = matrix_deflation(x_array, y_array, x_array, y_array, xw[:,0:1], yw[:,0:1])

# plot
for i in range(NVECS):
    print('\nCOMPONENT %i' % i)
    plt.scatter(np.arange(xw.shape[0]), xw[:,i]*100)
    plt.show()
    plt.scatter(np.arange(yw.shape[0]), yw[:,i]*100)
    plt.show()
    print('\n')

np.set_printoptions(suppress=True)
print('\n')
# covariates
xproj = np.dot(x_array, xw)
yproj = np.dot(y_array, yw)
covar = np.dot(xproj.T, yproj)
print(np.round(covar, 2))

for i in range(1,NVECS):
    for j in range(i):
        print(scipy.stats.pearsonr(xproj[:,i],yproj[:,j])[0])