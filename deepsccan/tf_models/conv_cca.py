"""
Convolutional CCA
"""

import numpy as np
np.set_printoptions(suppress=True)
import scipy.stats
import matplotlib.pyplot as plt

import tensorflow as tf


def conv2d_layer(x, filters, kernel_size, stride, activation, 
                 sparsity=0., nonneg=False, name='conv'):
    if sparsity > 0:
        regularizer = l1_regularizer(l1_penalty=sparsity)
    else:
        regularizer = None

    y = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, 
                    strides=stride, padding='same', activation=activation,
                    kernel_regularizer=regularizer, name=name)
    
    clip_conv = False
    if clip_conv:
        # create clip norm function
        with tf.variable_scope(name, reuse=True):
            weights = tf.get_variable('kernel')
            orthogonalized_weights = weights * tf.rsqrt(tf.reduce_sum(tf.square(y), axis=0))
            ortho_weights = tf.assign(weights, orthogonalized_weights, name='ortho')
            tf.add_to_collection('normalize_ops', ortho_weights)
            if nonneg:
                nonneg_op = tf.assign(weights, tf.maximum(0., weights))
                tf.add_to_collection('normalize_ops', nonneg_op)


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


class ConvCCA(object):

    def __init__(self,
                 x_conv,
                 x_hidden,
                 y_conv,
                 y_hidden,
                 nvecs,
                 activation='linear',
                 sparsity=0.,
                 nonneg=False,
                 deflation=True):
        self._x_conv = x_conv
        self._x_hidden = x_hidden
        self._y_conv = y_conv
        self._y_hidden = y_hidden
        self.nvecs = nvecs
        self.activation = self._parse_activation(activation)

        ## sparsity -> L1 penalty
        if not isinstance(sparsity, list) and not isinstance(sparsity, tuple):
            sparsity = [sparsity, sparsity]
        elif isinstance(sparsity, tuple):
            sparsity = list(sparsity)
        self.sparsity = sparsity

        self._nonneg = nonneg
        self._deflation = deflation
        tf.reset_default_graph()
        self._sess = tf.Session()

    def _parse_activation(self, act_input):
        if not isinstance(act_input, str):
            act_output = act_input 
        else:
            act_input = act_input.upper()
            if act_input == 'LINEAR':
                act_output = None
            elif act_input == 'SIGMOID':
                act_output = tf.nn.sigmoid
            elif act_input == 'RELU':
                act_output = tf.nn.relu
            elif act_input == 'ELU':
                act_output = tf.nn.elu
        return act_output

    def _process_inputs(self, x_array, y_array):
        if x_array.ndim == 3:
            x_array = np.expand_dims(x_array, -1)
        if y_array.ndim == 3:
            y_array = np.expand_dims(y_array, -1)
        return x_array, y_array

    def _inference(self, x_place, y_place):
        """
        Forward pass of the model
        """
        ### X PROJECTION
        ## conv layers
        xc_vals = self._x_conv[0]
        x_proj = conv2d_layer(x_place, xc_vals[0], xc_vals[1], xc_vals[2], 
                        sparsity=self.sparsity[0], padding='same', activation=self.activation,
                        name='x_conv_0')
        for idx, xc_vals in enumerate(self._x_conv[1:]):
            x_proj = conv2d_layer(x_place, xc_vals[0], xc_vals[1], xc_vals[2], 
                        sparsity=self.sparsity[0], padding='same', activation=self.activation,
                        name='x_conv_%i'% (idx+1))
        ## dense layers
        for idx, x_units in enumerate(self._x_hidden):
            x_proj = dense_layer(x_proj, units=x_units, activation=self.activation,
                        sparsity=self.sparsity[0], nonneg=self._nonneg, name='x_dense_%i' % idx)

        ### Y PROJECTION
        ## conv layers
        yc_vals = self._y_conv[0]
        y_proj = conv2d_layer(y_place, yc_vals[0], yc_vals[1], yc_vals[2], 
                        sparsity=self.sparsity[1], padding='same', activation=self.activation,
                        name='y_conv_0')
        for idx, yc_vals in enumerate(self._y_conv[1:]):
            y_proj = conv2d_layer(y_place, yc_vals[0], yc_vals[1], yc_vals[2], 
                        sparsity=self.sparsity[1], padding='same', activation=self.activation,
                        name='y_conv_%i'% (idx+1))
        ## dense layers
        for idx, y_units in enumerate(self._y_hidden):
            y_proj = dense_layer(y_proj, units=y_units, activation=self.activation,
                        sparsity=self.sparsity[1], nonneg=self._nonneg, name='y_dense_%i' % idx)

        return x_proj, y_proj

    def _loss(self, x_proj, y_proj):
        """
        upper_loss = tf.reduce_sum(tf.matrix_band_part(tf.abs(covar_mat), 0, -1))
        lower_loss = tf.reduce_sum(tf.matrix_band_part(tf.abs(covar_mat), -1, 0))
        """
        # projection covariance matrix
        covar_mat = tf.abs(tf.matmul(tf.transpose(x_proj), y_proj))

        # upper triangle sum
        ux,uy = np.triu_indices(self.nvecs, k=1)
        u_idxs = [[aa,bb] for aa,bb in zip(ux,uy)]
        upper_loss = tf.reduce_sum(tf.gather_nd(covar_mat, u_idxs))

        # lower triangle sum
        lx, ly = np.tril_indices(self.nvecs, k=-1)
        l_idxs = [[aa,bb] for aa,bb in zip(lx,ly)]
        lower_loss = tf.reduce_sum(tf.gather_nd(covar_mat, l_idxs))

        # diagonal sum
        diag_loss = tf.reduce_sum(tf.diag_part(covar_mat))

        total_loss = diag_loss + lower_loss + upper_loss

        if self.sparsity[0] > 0. or self.sparsity[1] > 0.:
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = tf.add(total_loss, tf.add_n(reg_losses))

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


    def fit(self, x, y, nb_epoch=1000, batch_size=32, learn_rate=1e-4):
        x_array, y_array = self._process_inputs(x, y)

        x_place = tf.placeholder(tf.float32, shape=(None, x_array.shape[-1]), name='x_place')
        y_place = tf.placeholder(tf.float32, shape=(None, y_array.shape[-1]), name='y_place')

        x_proj, y_proj = self._inference(x_place, y_place)
        total_loss = self._loss(x_proj, y_proj)
        train_op = self._train_op(total_loss, learn_rate)

        normalize_ops = tf.get_collection('normalize_ops')

        self.x_components = np.zeros((x_array.shape[-1], self.nvecs))
        self.y_components = np.zeros((y_array.shape[-1], self.nvecs))

        nb_batches = int(np.ceil(x_array.shape[0]/batch_size))

        self._sess.run(tf.global_variables_initializer())

        for epoch in range(nb_epoch):
            e_loss = np.zeros(nb_batches)
            for batch_idx in range(nb_batches):
                x_batch = x_array[batch_idx*batch_size:(batch_idx+1)*batch_size]
                y_batch = y_array[batch_idx*batch_size:(batch_idx+1)*batch_size]

                t_loss, _ = self._sess.run([total_loss, train_op],
                                                    feed_dict={x_place:x_batch,
                                                               y_place:y_batch})
                #print('Loss %.02f' % (loss))
                e_loss[batch_idx] = t_loss
                self._sess.run(normalize_ops,feed_dict={x_place:x_batch,
                                                        y_place:y_batch})

            print('Epoch Loss: %.03f' % np.mean(e_loss))