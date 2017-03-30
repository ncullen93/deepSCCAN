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
        x_proj = time_distributed_dense(x, units=self.nvecs,
                                        activation=self.activation,
                                        sparsity=self.sparse_x,
                                        smoothness=self.smooth_x,
                                        name='x_proj')
        y_proj = time_distributed_dense(y, units=self.nvecs,
                                        activation=self.activation,
                                        sparsity=self.sparse_y,
                                        smoothness=self.smooth_y,
                                        name='y_proj')

        return x_proj, y_proj

    def _loss(self, x_proj, y_proj):
        """
        Unconstrained mininimzation
        """
        covar_mat = tf.matmul(tf.transpose(x_proj), y_proj)
        cca_score = tf.reduce_sum(tf.diag_part(covar_mat))
        #diag_sum = 2. * tf.reduce_mean(tf.diag_part(covar_mat))
        #inter_sum = tf.reduce_mean(tf.matrix_band_part(covar_mat, 0, -1))
        #cca_score = diag_sum - inter_sum
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        return tf.add(cca_score, tf.add_n(reg_losses))

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
        corr_vals = []
        update_ops = []
        for i in range(self.nvecs):
            corr_val, update_op = metrics.streaming_pearson_correlation(x_proj[:,i], y_proj[:,i])
            corr_vals.append(corr_val)
            update_ops.append(update_op)
        #corr_op = tf.group(*corr_vals)
        update_op = tf.group(*update_ops)
        return corr_vals, update_op

    def fit(self, x, y, nb_epoch=100, batch_size=16,
            optimizer='adam', learn_rate=1e-3, verbose=True):
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
        corr_vals, eval_update_op = self._eval_op(x_proj, y_proj)

        # ops to clip weights so they have unit norm
        maxnorm_ops = tf.get_collection("maxnorm")

        nb_batches = int(np.ceil(x_array.shape[1] / float(batch_size)))
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

        for epoch in range(nb_epoch):
            x_batches = np.array_split(x_array, nb_batches, axis=1)
            y_batches = np.array_split(y_array, nb_batches, axis=1)
            for x_batch, y_batch in zip(x_batches, y_batches):
                batch_loss, _, _ = self.sess.run([loss, train_op, eval_update_op],
                                     feed_dict={x_place: x_batch,
                                                y_place: y_batch})
                # clip max norm of weights
                self.sess.run(maxnorm_ops)
                # update corr values after weight updates
                self.sess.run(eval_update_op)

                print('Loss: %.02f' % (batch_loss))
                c_vals = [cca_model.sess.run(c) for c in corr_vals]
                print('Avg Comp. Corr: %.02f' % np.mean([c_vals]))
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
                                       feed_dict={self.x_place: x,
                                                  self.y_place: y})

        return x_proj, y_proj

    def __del__(self):
        self.sess.close()


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
        return tf.add_n([l1_score, l2_score])
    return l1l2


def maxnorm_l1l2_regularizer(threshold, axes, l1_penalty, l2_penalty):
    def maxnorm_l1l2(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name='maxnorm')
        tf.add_to_collection('maxnorm', clip_weights)
        l1_score = tf.multiply(tf.reduce_sum(tf.abs(weights)), 
                        tf.convert_to_tensor(l1_penalty, dtype=tf.float32))
        l2_score = tf.multiply(tf.nn.l2_loss(weights),
                               tf.convert_to_tensor(l2_penalty, dtype=tf.float32))
        final_score = tf.add_n([l1_score, l2_score])
        #tf.add_to_collection('losses', final_score)
        return final_score
    return maxnorm_l1l2

def time_distributed_dense(x, units, activation, sparsity, smoothness, name):
    """
    Layer that applies a dense matrix multiplication across
    all the samples as if they were one sample
    """
    #input_shape = x.shape.as_list()
    #x = tf.reshape(x, [-1, input_shape[2]])
    y = tf.layers.dense(x, units=units, activation=activation,
            kernel_regularizer=maxnorm_l1l2_regularizer(1.0, 0, sparsity, smoothness),
            name=name)
    #y = tf.reshape(y, [1, -1, units])
    return y


def cca_score_layer(x_proj, y_proj):
    """
    Calculates the CCA objective value from the
    two projections
    """
    output = tf.matmul(x_proj, y_proj)
    output_square = tf.square(output)
    output_sum = tf.reduce_sum(output_square)
    return output_sum


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