"""
Standard CCA model which learns components one-by-one,
applying matrix deflation after each component.
"""

import numpy as np
np.set_printoptions(suppress=True)
import scipy.stats
import matplotlib.pyplot as plt

import tensorflow as tf

def dense_layer(x, units, sparsity=0., smoothness=0., name='dense'):
    if sparsity > 0. or smoothness > 0.:
        regularizer = l1_regularizer(l1_penalty=sparsity)
    else:
        regularizer = None
    
    y = tf.layers.dense(x, units=units, activation=None,
            kernel_regularizer=regularizer, use_bias=False, name=name)

    # create clip norm function
    with tf.variable_scope(name, reuse=True):
        weights = tf.get_variable('kernel')
        orthogonalized_weights = weights * tf.rsqrt(tf.reduce_sum(tf.square(y), axis=0))
        ortho_weights = tf.assign(weights, orthogonalized_weights, name='ortho')
        tf.add_to_collection('normalize_ops', ortho_weights)
        #weights = tf.get_variable('kernel')
        #clipped = tf.clip_by_norm(weights, clip_norm=1.0, axes=0)
        #clip_weights = tf.assign(weights, clipped, name='ortho')
        #tf.add_to_collection('normalize_ops', clip_weights)

    return y


def l1_regularizer(l1_penalty):
    def l1(weights):
        l1_score = tf.multiply(tf.reduce_sum(tf.abs(weights)), 
                        tf.convert_to_tensor(l1_penalty, dtype=tf.float32))
        return l1_score
    return l1

def l1l2_regularizer(l1_penalty, l2_penalty):
    def l1l2(weights):
        l1_score = tf.multiply(l1_penalty, tf.reduce_sum(tf.abs(weights)))
        l2_score = tf.multiply(l2_penalty, tf.nn.l2_loss(weights))
        return tf.add(l1_score, l2_score)
    return l1l2


def matrix_deflation(X_curr, Y_curr, X_orig, Y_orig, u, v):
    """
    Deflate matrices and return new ones
    """
    Xp = X_curr
    Yp = Y_curr
    #u      = u / (np.sqrt(np.sum(u**2)+1e-7))
    #v      = v / (np.sqrt(np.sum(v**2)+1e-7))

    qx = np.dot(Xp.T,np.dot(X_orig,u))
    qx = qx / (np.sqrt(np.sum(qx**2)+1e-7))
    Xp = Xp - np.dot(Xp,qx).dot(qx.T)

    qy = np.dot(Yp.T,np.dot(Y_orig,v))
    qy = qy / (np.sqrt(np.sum(qy**2)+1e-7))
    Yp = Yp - np.dot(Yp,qy).dot(qy.T)

    Xp = (Xp - np.mean(Xp)) / np.std(Xp)
    Yp = (Yp - np.mean(Yp)) / np.std(Yp)

    return Xp, Yp


class CCA(object):

    def __init__(self, 
                 nvecs, 
                 sparsity=0., 
                 smoothness=0., 
                 deflation=True):
        self.nvecs = nvecs

        ## sparsity -> L1 penalty
        if not isinstance(sparsity, list) and not isinstance(sparsity, tuple):
            sparsity = [sparsity, sparsity]
        elif isinstance(sparsity, tuple):
            sparsity = list(sparsity)
        self._frac_sparsity = sparsity
        self.sparsity = [0., 0.]

        ## smoothness -> L2 Penalty
        if not isinstance(smoothness, list) and not isinstance(smoothness, tuple):
            smoothness = [smoothness, smoothness]
        elif isinstance(smoothness, tuple):
            smoothness = list(smoothness)
        self._frac_smoothness = smoothness
        self.smoothness = [0., 0.]
        
        self._deflation = deflation
        tf.reset_default_graph()
        self._sess = tf.Session()

    def _process_inputs(self, x_array, y_array):
        x_array = x_array.reshape(x_array.shape[0], -1)
        y_array = y_array.reshape(y_array.shape[0], -1)
        return x_array, y_array

    def _inference(self, x_place, y_place):
        if self._deflation:
            units = 1
        else:
            units = self.nvecs
        x_proj = dense_layer(x_place, units=units, sparsity=self.sparsity[0], 
                             smoothness=self.smoothness[0], name='x_proj')
        y_proj = dense_layer(y_place, units=units, sparsity=self.sparsity[1], 
                             smoothness=self.smoothness[0], name='y_proj')
        return x_proj, y_proj     

    def _loss(self, x_proj, y_proj):
        covar_mat = tf.matmul(tf.transpose(x_proj), y_proj)
        cca_loss = tf.reduce_sum(tf.diag_part(tf.abs(covar_mat)))
        #if not self._deflation:
        upper_loss = tf.reduce_sum(tf.matrix_band_part(tf.abs(covar_mat), 0, -1))
        lower_loss = tf.reduce_sum(tf.matrix_band_part(tf.abs(covar_mat), -1, 0))
        total_cca_loss = -3.*cca_loss + upper_loss + lower_loss

        #if self.sparsity[0] > 0. or self.sparsity[1] > 0.:
        #    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #    total_loss = tf.add(total_cca_loss, tf.add_n(reg_losses))
        #else:
        return total_cca_loss, cca_loss

    def _train_op(self, loss, learn_rate, clip_value=3.):
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
        #optimizer = tf.train.MomentumOptimizer(learning_rate=learn_rate, momentum=0.9)
        #gvs = optimizer.compute_gradients(loss)
        #clipped_gvs = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in gvs]
        #train_op = optimizer.apply_gradients(clipped_gvs)
        train_op = optimizer.minimize(loss)
        return train_op

    def fit(self, x, y, nb_epoch=1000, batch_size=32, learn_rate=1e-4):
        x_array, y_array = self._process_inputs(x, y)
        self.sparsity[0] = self._frac_sparsity[0]# * np.sqrt(x_array.shape[-1])
        self.sparsity[1] = self._frac_sparsity[1]# * np.sqrt(y_array.shape[-1])

        x_place = tf.placeholder(tf.float32, shape=(None, x_array.shape[-1]), name='x_place')
        y_place = tf.placeholder(tf.float32, shape=(None, y_array.shape[-1]), name='y_place')

        x_proj, y_proj = self._inference(x_place, y_place)
        total_loss, cca_loss = self._loss(x_proj, y_proj)
        train_op = self._train_op(total_loss, learn_rate)

        normalize_ops = tf.get_collection('normalize_ops')

        self.x_components = np.zeros((x_array.shape[-1], self.nvecs))
        self.y_components = np.zeros((y_array.shape[-1], self.nvecs))

        nb_batches = int(np.ceil(x_array.shape[0]/batch_size))

        self._sess.run(tf.global_variables_initializer())

        if self._deflation:
            x_array_orig = x_array.copy()
            y_array_orig = y_array.copy()
            component_loops = self.nvecs
        else:
            component_loops = 1

        for c_idx in range(component_loops):
            self._sess.run(tf.global_variables_initializer())

            for epoch in range(nb_epoch):
                e_loss = np.zeros(nb_batches)
                for batch_idx in range(nb_batches):
                    x_batch = x_array[batch_idx*batch_size:(batch_idx+1)*batch_size]
                    y_batch = y_array[batch_idx*batch_size:(batch_idx+1)*batch_size]

                    t_loss, c_loss, _ = self._sess.run([total_loss, cca_loss, train_op],
                                                        feed_dict={x_place:x_batch,
                                                                   y_place:y_batch})
                    self._sess.run(normalize_ops, feed_dict={x_place:x_batch, y_place:y_batch})
                    #print('Loss %.02f' % (loss))
                    e_loss[batch_idx] = t_loss

               
                print('Component Correlation: %.03f' % np.mean(e_loss))

            ## get components and projects
            with tf.variable_scope('', reuse=True):
                xw = self._sess.run(tf.get_variable('x_proj/kernel'))
                yw = self._sess.run(tf.get_variable('y_proj/kernel'))
                xw = xw / np.sqrt(np.sum(xw**2) + 1e-12)
                yw = yw / np.sqrt(np.sum(yw**2) + 1e-12)
                if self._deflation:
                    self.x_components[:,c_idx] = xw.flatten()
                    self.y_components[:,c_idx] = yw.flatten()
                else:
                    self.x_components = xw
                    self.y_components = yw

            ## matrix deflation if necessary
            if self._deflation:
                x_array, y_array = matrix_deflation(x_array, y_array,
                                        x_array_orig, y_array_orig, xw, yw)


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
    #X = (X - np.mean(X)) / np.std(X)
    Y = np.dot(v2+e2, u.T)
    #Y = (Y - np.mean(Y)) / np.std(Y)

    return X.T, Y.T, v1, v2, u

if __name__ == '__main__':
    EXAMPLE = 'MNIST' # SYNTHETIC, FACES, MNIST, MNIST_SINGLE BRAIN

    if EXAMPLE == 'SYNTHETIC':
        x_array, y_array, v1, v2, u = generate_data()

        model = CCA(nvecs=2, sparsity=(0.5, 0.5), deflation=False)
        model.fit(x_array, y_array, nb_epoch=500, batch_size=500, learn_rate=1e-4)

        xw, yw = model.x_components, model.y_components

        for i in range(xw.shape[-1]):
            plt.scatter(np.arange(xw.shape[0]), xw[:,i]*100)
            plt.show()
            plt.scatter(np.arange(yw.shape[0]), yw[:,i]*100)
            plt.show()

        xproj = np.dot(x_array, xw)
        yproj = np.dot(y_array, yw)
        for i in range(xproj.shape[-1]):
            print(scipy.stats.pearsonr(xproj[:,i],yproj[:,i])[0])

    elif EXAMPLE == 'FACES':
        from sklearn.datasets import fetch_olivetti_faces
        data = fetch_olivetti_faces()
        images = data.images
        x = images[:,:,:32]
        y = images[:,:,32:]
        nvecs = 8
        model = CCA(nvecs=nvecs, sparsity=(0.05,0.05), smoothness=(0.1, 0.1),
                    deflation=False)
        model.fit(x, y, nb_epoch=300, batch_size=40, learn_rate=1e-4)
        
        xw, yw = model.x_components, model.y_components

        xw = xw.reshape(64,32,nvecs)
        yw = yw.reshape(64,32,nvecs)
        for i in range(nvecs):
            plt.imshow(xw[:,:,i])
            plt.show()
            plt.imshow(yw[:,:,i])
            plt.show()
            print('\n\n')

    elif EXAMPLE == 'MNIST':
        from keras.datasets import mnist
        (xtrain,ytrain),(xtest,ytest) = mnist.load_data()
        xtrain = xtrain / 255.
        x = xtrain[:5000,:,:14]
        y = xtrain[:5000,:,14:]

        nvecs = 5
        model = CCA(nvecs=nvecs, sparsity=(0.001,0.001), smoothness=(0.1,0.1), deflation=True)
        model.fit(x, y, nb_epoch=100, batch_size=100, learn_rate=1e-3)

        xw, yw = model.x_components, model.y_components
        xw = xw.reshape(28,14,nvecs)
        yw = yw.reshape(28,14,nvecs)
        for i in range(nvecs):
            plt.imshow(xw[:,:,i])
            plt.show()
            plt.imshow(yw[:,:,i])
            plt.show()
            print('\n\n')

        xw, yw = model.x_components, model.y_components
        x = x.reshape(x.shape[0],-1)
        y = y.reshape(y.shape[0],-1)
        xproj = np.dot(x,xw)
        yproj = np.dot(y, yw)
        for i in range(nvecs):
            print(scipy.stats.pearsonr(xproj[:,i],yproj[:,i])[0])

        print(np.round(np.dot(xproj.T,yproj),2))

    elif EXAMPLE == 'MNIST_SINGLE':
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

        nvecs = 5
        model = CCA(nvecs=nvecs, sparsity=(0.01,0.01), smoothness=(0.1,0.1), deflation=False)
        model.fit(x, y, nb_epoch=300, batch_size=200, learn_rate=1e-4)
        xw, yw = model.x_components, model.y_components
        xw = xw.reshape(28,28,nvecs)
        for i in range(nvecs):
            plt.imshow(xw[:,:,i])
            plt.show()
            print('\n\n')

        xw, yw = model.x_components, model.y_components
        x = x.reshape(x.shape[0],-1)
        xproj = np.dot(x,xw)
        yproj = np.dot(y, yw)
        for i in range(nvecs):
            print(scipy.stats.pearsonr(xproj[:,i],yproj[:,i])[0])

        print(np.round(np.dot(xproj.T,yproj),2))