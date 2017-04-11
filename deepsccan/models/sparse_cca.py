"""
Standard CCA model which learns components one-by-one,
applying matrix deflation after each component.
"""

import numpy as np
np.set_printoptions(suppress=True)
import scipy.stats
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator

import tensorflow as tf


def dense_layer(x, units, activation=None, sparsity=0., nonneg=False, name='dense'):
    """
    Unit projection clipping (for real correlation maximization):
        weights = tf.get_variable('kernel')
        clipped = tf.clip_by_norm(weights, clip_norm=1.0, axes=0)
        clip_weights = tf.assign(weights, clipped, name='ortho')
        tf.add_to_collection('normalize_ops', clip_weights)
    """
    if sparsity > 0:
        regularizer = l1_regularizer(l1_penalty=sparsity)
    else:
        regularizer = None
    
    y = tf.layers.dense(x, units=units, activation=activation,
            kernel_regularizer=regularizer, use_bias=False, name=name)

    # create clip norm function
    with tf.variable_scope(name, reuse=True):
        weights = tf.get_variable('kernel')
        orthogonalized_weights = weights * tf.rsqrt(tf.reduce_sum(tf.square(y), axis=0))
        ortho_weights = tf.assign(weights, orthogonalized_weights, name='ortho')
        tf.add_to_collection('normalize_ops', ortho_weights)
        if nonneg:
            nonneg_op = tf.assign(weights, tf.maximum(0., weights))
            tf.add_to_collection('normalize_ops', nonneg_op)
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


def projection_correlations(x, y):
    corr_vals = [scipy.stats.pearsonr(x[:,i], y[:,i])[0] for i in range(x.shape[-1])]
    return corr_vals


def matrix_deflation(X_curr, Y_curr, X_orig, Y_orig, u, v):
    """
    Deflate matrices and return new ones
    """
    Xp = X_curr
    Yp = Y_curr

    qx = np.dot(Xp.T,np.dot(X_orig,u))
    qx = qx / (np.sqrt(np.sum(qx**2)+1e-7))
    Xp = Xp - np.dot(Xp,qx).dot(qx.T)

    qy = np.dot(Yp.T,np.dot(Y_orig,v))
    qy = qy / (np.sqrt(np.sum(qy**2)+1e-7))
    Yp = Yp - np.dot(Yp,qy).dot(qy.T)

    Xp = (Xp - np.mean(Xp)) / np.std(Xp)
    Yp = (Yp - np.mean(Yp)) / np.std(Yp)

    return Xp, Yp


class SparseCCA(BaseEstimator):

    def __init__(self,
                 ncomponents=10,
                 activation='linear',
                 sparsity=0.,
                 nonneg=False,
                 deflation=True,
                 nb_epoch=1000, 
                 batch_size=32, 
                 learn_rate=1e-4,
                 device='/cpu:0',
                 verbose=False,
                 log_device=False):
        self.ncomponents = ncomponents
        self.activation = activation
        self.sparsity = sparsity
        self.nonneg = nonneg
        self.deflation = deflation
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.device = device
        self.verbose = verbose
        self.log_device = log_device

    def _parse_activation(self, act_input):
        activation_map = {
            'LINEAR' : None,
            'SIGMOID' : tf.nn.sigmoid,
            'RELU' : tf.nn.relu,
            'ELU' : tf.nn.elu
        }
        if not isinstance(act_input, str):
            act_output = act_input 
        else:
            try:
                act_output = activation_map[act_input.upper()]
            except:
                act_output = None

        return act_output

    def _process_inputs(self, x_array, y_array):
        x_array = x_array.reshape(x_array.shape[0], -1)
        y_array = y_array.reshape(y_array.shape[0], -1)
        return x_array, y_array

    def _inference(self, x_place, y_place):
        if self.deflation:
            units = 1
        else:
            units = self.ncomponents
        with tf.device(self.device):
            x_proj = dense_layer(x_place, units=units, sparsity=self.sparsity[0], 
                                 nonneg=self.nonneg, name='x_proj')
            y_proj = dense_layer(y_place, units=units, sparsity=self.sparsity[1], 
                                 nonneg=self.nonneg, name='y_proj')
        return x_proj, y_proj

    def _loss(self, x_proj, y_proj):
        """
        upper_loss = tf.reduce_sum(tf.matrix_band_part(tf.abs(covar_mat), 0, -1))
        lower_loss = tf.reduce_sum(tf.matrix_band_part(tf.abs(covar_mat), -1, 0))
        """
        # projection covariance matrix
        with tf.device(self.device):
            covar_mat = tf.abs(tf.matmul(tf.transpose(x_proj), y_proj))

            if not self.deflation:
                # diagonal sum
                diag_loss = tf.reduce_sum(tf.diag_part(covar_mat))

                # upper triangle sum
                ux,uy = np.triu_indices(self.ncomponents, k=1)
                u_idxs = [[aa,bb] for aa,bb in zip(ux,uy)]
                upper_loss = tf.reduce_sum(tf.gather_nd(covar_mat, u_idxs))

                # lower triangle sum
                lx, ly = np.tril_indices(self.ncomponents, k=-1)
                l_idxs = [[aa,bb] for aa,bb in zip(lx,ly)]
                lower_loss = tf.reduce_sum(tf.gather_nd(covar_mat, l_idxs))

                total_loss = -1.*diag_loss + lower_loss + upper_loss
            else:
                total_loss = -tf.reduce_sum(covar_mat)

            if self.sparsity[0] > 0. or self.sparsity[1] > 0.:
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                total_loss = total_loss + tf.add_n(reg_losses)

        return total_loss

    def _train_op(self, loss, learn_rate):
        """
        Clipping:
        #gvs = optimizer.compute_gradients(loss)
        #clipped_gvs = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in gvs]
        #train_op = optimizer.apply_gradients(clipped_gvs)
        """      
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
        train_op = optimizer.minimize(loss)
        return train_op

    def fit(self, X, y, nb_epoch=None, batch_size=None):
        if nb_epoch is not None:
            self.nb_epoch = nb_epoch
        if batch_size is not None:
            self.batch_size = batch_size

        if self.deflation:
            self.nb_epoch = int(self.nb_epoch / self.ncomponents)
        ## parameter validation
        self.activation = self._parse_activation(self.activation)
        if not isinstance(self.sparsity, list) and not isinstance(self.sparsity, tuple):
            self.sparsity = [self.sparsity, self.sparsity]
        elif isinstance(self.sparsity, tuple):
            self.sparsity = list(self.sparsity)
        self.sparsity = self.sparsity

        # process inputs
        x_array, y_array = self._process_inputs(X, y)

        # create isolated tensorflow graph to hold this model's tensors/ops
        self._graph = tf.Graph()

        with self._graph.as_default():
            x_place = tf.placeholder(tf.float32, shape=(None, x_array.shape[-1]), 
                name='x_place')
            y_place = tf.placeholder(tf.float32, shape=(None, y_array.shape[-1]), 
                name='y_place')

            x_proj, y_proj = self._inference(x_place, y_place)
            total_loss = self._loss(x_proj, y_proj)
            train_op = self._train_op(total_loss, self.learn_rate)

            normalize_ops = tf.get_collection('normalize_ops')
            init_op = tf.global_variables_initializer()

            nb_batches = int(np.ceil(x_array.shape[0]/self.batch_size))

            # create tensorflow session
            self._sess = tf.Session(graph=self._graph,
                config=tf.ConfigProto(log_device_placement=self.log_device))
            # initialize global variables
            self._sess.run(init_op)

            if self.deflation:
                x_array_orig = x_array.copy()
                y_array_orig = y_array.copy()
                component_loops = self.ncomponents
            else:
                x_array_orig = x_array
                y_array_orig = y_array
                component_loops = 1

            for c_idx in range(component_loops):
                self._sess.run(tf.global_variables_initializer())

                for epoch in range(self.nb_epoch):
                    e_loss = np.zeros(nb_batches)
                    for batch_idx in range(nb_batches):
                        x_batch = x_array[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                        y_batch = y_array[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]

                        t_loss, _ = self._sess.run([total_loss, train_op],
                                                            feed_dict={x_place:x_batch,
                                                                       y_place:y_batch})
                        #print('Loss %.02f' % (loss))
                        e_loss[batch_idx] = t_loss
                        self._sess.run(normalize_ops,feed_dict={x_place:x_batch,
                                                                y_place:y_batch})

                    if self.verbose:
                        print('Epoch Loss: %.03f' % np.mean(e_loss))

                ## get components and projects
                with tf.variable_scope('', reuse=True):
                    xw = self._sess.run(tf.get_variable('x_proj/kernel'))
                    yw = self._sess.run(tf.get_variable('y_proj/kernel'))
                    xw = xw / np.sqrt(np.sum(xw**2) + 1e-12)
                    yw = yw / np.sqrt(np.sum(yw**2) + 1e-12)
                    if self.deflation:
                        if c_idx == 0:
                            self.x_components_ = np.zeros((x_array.shape[-1], 
                                                           self.ncomponents))
                            self.y_components_ = np.zeros((y_array.shape[-1], 
                                                           self.ncomponents))
                        self.x_components_[:,c_idx] = xw.flatten()
                        self.y_components_[:,c_idx] = yw.flatten()
                    else:
                        self.x_components_ = xw
                        self.y_components_ = yw
                x_proj = np.dot(x_array_orig, xw)
                y_proj = np.dot(y_array_orig, yw)
                corr_val = projection_correlations(x_proj, y_proj)
                if self.verbose:
                    print('Corrs: ' , corr_val)
                ## matrix deflation if necessary
                if self.deflation:
                    x_array, y_array = matrix_deflation(x_array, y_array,
                                            x_array_orig, y_array_orig, xw, yw)
            self._sess.close()
            return self

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        x_array, y_array = self._process_inputs(X, y)
        x_proj = np.dot(x_array, self.x_components_)
        y_proj = np.dot(y_array, self.y_components_)
        corr_vals = projection_correlations(x_proj, y_proj)
        return corr_vals

    def score(self, X, y):
        return np.sum(np.abs(self.evaluate(X, y)))

    def transform(self, X, y=None):
        x_proj = np.dot(X, self.x_components_)
        if y is not None:
            y_proj = np.dot(y, self.y_components_)
            return (x_proj, y_proj)
        else:
            return x_proj

    def inverse_transform(self, X):
        y_proj = np.dot(X, self.y_components_)
        return y_proj


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
    EXAMPLE = None # SYNTHETIC, FACES, MNIST, MNIST_SINGLE BRAIN

    if EXAMPLE == 'SYNTHETIC':
        x_array, y_array, v1, v2, u = generate_data()

        model = SparseCCA(ncomponents=2, sparsity=(0.5,0.5), nonneg=False, deflation=False)
        model.fit(x_array, y_array, nb_epoch=2000, batch_size=500, learn_rate=1e-5)

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
        print(np.round(np.dot(xproj.T, yproj)))

    elif EXAMPLE == 'FACES':
        from sklearn.datasets import fetch_olivetti_faces
        data = fetch_olivetti_faces()
        images = data.images
        x = images[:,:,:32]
        y = images[:,:,32:]
        ncomponents = 8
        model = SparseCCA(ncomponents=ncomponents, sparsity=(0.05,0.05), deflation=False)
        model.fit(x, y, nb_epoch=300, batch_size=40, learn_rate=1e-4)
        
        xw, yw = model.x_components, model.y_components

        xw = xw.reshape(64,32,ncomponents)
        yw = yw.reshape(64,32,ncomponents)
        for i in range(ncomponents):
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

        ncomponents = 5
        model = SparseCCA(ncomponents=ncomponents, sparsity=(0.01,0.01), deflation=False)
        model.fit(x, y, nb_epoch=300, batch_size=64, learn_rate=1e-4)

        xw, yw = model.x_components, model.y_components
        xw = xw.reshape(28,14,ncomponents)
        yw = yw.reshape(28,14,ncomponents)
        #for i in range(ncomponents):
        #    plt.imshow(xw[:,:,i])
        #    plt.show()
        #    plt.imshow(yw[:,:,i])
        #    plt.show()
        #    print('\n\n')

        xw, yw = model.x_components, model.y_components
        x = x.reshape(x.shape[0],-1)
        y = y.reshape(y.shape[0],-1)
        xproj = np.dot(x,xw)
        yproj = np.dot(y, yw)
        for i in range(ncomponents):
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

        ncomponents = 5
        model = SparseCCA(ncomponents=ncomponents, sparsity=100., deflation=False)
        model.fit(x, y, nb_epoch=300, batch_size=200, learn_rate=1e-4)
        xw, yw = model.x_components, model.y_components
        xw = xw.reshape(28,28,ncomponents)
        for i in range(ncomponents):
            plt.imshow(xw[:,:,i])
            plt.show()
            print('\n\n')

        xw, yw = model.x_components, model.y_components
        x = x.reshape(x.shape[0],-1)
        xproj = np.dot(x,xw)
        yproj = np.dot(y, yw)
        for i in range(ncomponents):
            print(scipy.stats.pearsonr(xproj[:,i],yproj[:,i])[0])

        print(np.round(np.dot(xproj.T,yproj),2))