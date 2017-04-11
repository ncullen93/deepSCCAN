"""
Sparse CCA Model with Linear and Non-Linear Activations
"""

import tensorflow as tf
from tensorflow.contrib import metrics
import numpy as np


class SparseCCA(object):
    """
    Sparse CCA Model
    """

    def __init__(self,
                 nvecs,
                 activation='linear',
                 sparsity=(0., 0.),
                 smoothness=(0., 0.),
                 nonneg=False):
        """
        Initialize Sparse CCA Model

        Arguments
        ---------
        nvecs : integer
            number of components to learn

        sparsity : tuple of floats
            sparsity L1 penalty for X and Y data

        smoothness : tuple of floats
            smoothness L2 penalty for X and Y data

        nonneg : boolean
            whether all components should be non-negative

        optimizer : string
            optimization algorithm to use

        verbose : boolean
            whether to print verbose updates during training
        """
        tf.reset_default_graph()
        self.nvecs = nvecs
        self.activation = self.parse_activation(activation)
        self.sparse_x, self.sparse_y = sparsity
        self.smooth_x, self.smooth_y = smoothness
        self.nonneg = nonneg

        # create session
        self.sess = tf.Session()

    def process_inputs(self, x, y):
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        return x, y

    def parse_activation(self, value):
        if value.upper() == 'LINEAR':
            to_return = None
        elif value.upper() == 'RELU':
            to_return = tf.nn.relu
        elif value.upper() == 'ELU':
            to_return = tf.nn.elu
        else:
            to_return = None
        return to_return

    def _inference(self, x, y):
        """
        Notes
        -----
        - weights should be constrained unit norm
        """
        x_proj = dense_layer(x, units=self.nvecs,
                                        activation=self.activation,
                                        sparsity=self.sparse_x,
                                        smoothness=self.smooth_x,
                                        name='x_proj')
        y_proj = dense_layer(y, units=self.nvecs,
                                        activation=self.activation,
                                        sparsity=self.sparse_y,
                                        smoothness=self.smooth_y,
                                        name='y_proj')

        return x_proj, y_proj

    def _loss(self, x_proj, y_proj):
        """
        Value to MINIMIZE
        """
        covar_mat = tf.matmul(tf.transpose(x_proj), y_proj)
        diag_sum = tf.reduce_sum(tf.abs(tf.diag_part(covar_mat)))
        cca_score = tf.multiply(-1., diag_sum)
        #inter_sum = tf.reduce_sum(tf.abs(tf.matrix_band_part(covar_mat, 0, -1)))
        #cca_score = tf.multiply(-1., diag_sum - inter_sum) 
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = cca_score + tf.add_n(reg_losses)
        return total_loss

    def _train_op(self, loss, optimizer, learn_rate):
        """
        Returns training op from loss tensor
        """
        if optimizer.upper() == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate=learn_rate)
        elif optimizer.upper() == 'SGD':
            opt = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
        train_op = opt.minimize(loss)
        return train_op

    def _eval_op(self, x_proj, y_proj):
        """
        Returns evaluation op from x and y projections
        """
        corr_vals = tf_pearson_correlation(x_proj, y_proj)
        return corr_vals

    def fit(self, x, y, nb_epoch=100, batch_size=16,
            optimizer='adam', learn_rate=1e-5, verbose=True):
        """
        Fit the Sparse CCA model
        """
        # process inputs
        x_array, y_array = self.process_inputs(x, y)

        # create placeholders
        x_place = tf.placeholder(tf.float32, [None, x_array.shape[-1]],
                                 name='x_placeholder')
        y_place = tf.placeholder(tf.float32, [None, y_array.shape[-1]],
                                 name='y_placeholder')

        # create weight variables
        x_proj, y_proj = self._inference(x_place, y_place)
        loss = self._loss(x_proj, y_proj)
        train_op = self._train_op(loss, optimizer, learn_rate)
        corr_vals = self._eval_op(x_proj, y_proj)

        # ops to clip weights so they have unit norm
        maxnorm_ops = tf.get_collection("maxnorm")

        nb_batches = int(np.ceil(x_array.shape[1] / float(batch_size)))
        init_op = tf.global_variables_initializer()

        self.sess.run(init_op)

        for epoch in range(nb_epoch):
            for batch_idx in range(nb_batches):
                x_batch = x_array[batch_idx*batch_size:(batch_idx+1)*batch_size]
                y_batch = y_array[batch_idx*batch_size:(batch_idx+1)*batch_size]

                batch_loss, _ = self.sess.run([loss, train_op],
                                     feed_dict={x_place: x_batch,
                                                y_place: y_batch})
                # clip max norm of weights
                self.sess.run(maxnorm_ops)
                if verbose:
                    print('Loss: %.02f' % (batch_loss))

                if verbose and batch_idx % 10 == 0:
                    batch_corr_vals = self.sess.run(corr_vals,
                                            feed_dict={x_place:x_batch,
                                                       y_place: y_batch})
                    print('Corrs: ' , batch_corr_vals)

        # store variables for prediction & evaluation
        self.x_place = x_place
        self.y_place = y_place
        self.x_proj = x_proj
        self.y_proj = y_proj

    def predict(self, x, y):
        if self.x_place is None:
            raise Exception('Must fit model first!')

        x_array, y_array = self.process_inputs(x, y)

        x_proj, y_proj = self.sess.run([self.x_proj, self.y_proj],
                                       feed_dict={self.x_place: x_array,
                                                  self.y_place: y_array})

        return x_proj, y_proj

    def __del__(self):
        self.sess.close()


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

def maxnorm_regularizer(threshold, axes=0, name='maxnorm', collection='maxnorm'):
    def maxnorm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None
    return maxnorm


def l1l2_regularizer(l1_penalty, l2_penalty):
    def l1l2(weights):
        l1_score = tf.multiply(tf.reduce_sum(tf.abs(weights)), 
                        tf.convert_to_tensor(l1_penalty, dtype=tf.float32))
        l2_score = tf.multiply(tf.nn.l2_loss(weights),
                               tf.convert_to_tensor(l2_penalty, dtype=tf.float32))
        return tf.add(l1_score, l2_score)
    return l1l2

def dense_layer(x, units, activation, sparsity, smoothness, name):
    """
    Layer that applies a dense matrix multiplication across
    all the samples as if they were one sample
    """
    y = tf.layers.dense(x, units=units, activation=activation,
            kernel_regularizer=l1l2_regularizer(l1_penalty=sparsity, l2_penalty=smoothness),
            use_bias=False, name=name)
    # create clip norm function
    with tf.variable_scope(name, reuse=True):
        weights = tf.get_variable('kernel')
        clipped = tf.clip_by_norm(weights, clip_norm=1.0, axes=0)
        clip_weights = tf.assign(weights, clipped, name='maxnorm')
        tf.add_to_collection('maxnorm', clip_weights)
    return y


# TEST TIME-DISTRIBUTED-DENSE
if __name__ == '__main__':
    RUN_EXAMPLE = False
    if RUN_EXAMPLE:
        from keras.datasets import mnist

        (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
        xx = xtrain[:100]
        yy = xtrain[100:200]
        
        cca_model = SparseCCA(nvecs=10, activation='linear')
        cca_model.fit(xx, yy, nb_epoch=5, batch_size=16)