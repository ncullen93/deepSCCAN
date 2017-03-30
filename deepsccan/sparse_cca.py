"""
Sparse CCA Model with Linear and Non-Linear Activations
"""

import tensorflow as tf


class Test(object):

    def __init__(self):
        tf.reset_default_graph()

    def start(self):
        with tf.variable_scope('test'):
            x = tf.get_variable('nick', initializer=20.)
            x += 2
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(x))
        # self.x = x

    def stop(self):
        with tf.variable_scope('test', reuse=True):
            x = tf.get_variable('nick')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(x))


class SparseCCA(object):
    """
    Sparse CCA Model
    """

    def __init__(self,
                 nvecs,
                 activation='linear',
                 sparsity=(0, 0),
                 smoothness=(0, 0),
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
        # flatten inputs
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        # extra dimension for the single batch
        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)
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
        x_proj = time_distributed_dense_layer(x, units=self.nvecs,
                                              activation=self.activation,
                                              name='x_proj')
        y_proj = time_distributed_dense_layer(y, units=self.nvecs,
                                              activation=self.activation,
                                              name='y_proj')

        return x_proj, y_proj

    def _loss(self, x_proj, y_proj):
        """
        Unconstrained mininimzation
        """
        cca_score = tf.reduce_sum(
            tf.diag_part(
                tf.squeeze(
                    tf.matmul(
                        tf.transpose(tf.squeeze(x_proj)), tf.squeeze(y_proj)))))
        return -1 * cca_score

    def _training(self, loss, optimizer, learn_rate):
        """
        Returns training op from loss tensor
        """
        if optimizer.upper() == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate=learn_rate)
        elif optimizer.upper() == 'SGD':
            opt = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
        train_op = opt.minimize(loss)
        return train_op

    def evaluate(self, x, y):
        x_proj, y_proj = self.inference(x, y)

    def fit(self, x, y, nb_epoch=100, batch_size=16,
            optimizer='adam', learn_rate=1e-3, verbose=True):
        """
        Fit the Sparse CCA model
        """
        # process inputs
        x_array, y_array = self.process_inputs(x, y)

        # create placeholders
        x_place = tf.placeholder(tf.float32, [1, None, x_array.shape[-1]])
        y_place = tf.placeholder(tf.float32, [1, None, y_array.shape[-1]])

        # create weight variables
        x_proj, y_proj = self._inference(x_place, y_place)
        loss = self._loss(x_proj, y_proj)
        train_op = self._training(loss, optimizer, learn_rate)

        # ops to clip weights so they have unit norm
        maxnorm_ops = tf.get_collection("maxnorm")

        nb_batches = int(np.ceil(x_array.shape[1] / float(batch_size)))
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        for epoch in range(nb_epoch):
            x_batches = np.array_split(x_array, nb_batches, axis=1)
            y_batches = np.array_split(y_array, nb_batches, axis=1)
            for x_batch, y_batch in zip(x_batches, y_batches):
                _loss, _ = self.sess.run([loss, train_op],
                                     feed_dict={x_place: x_batch,
                                                y_place: y_batch})
                # clip max norm of weights
                self.sess.run(maxnorm_ops)
                print('Loss: %.02f' % (_loss))

    def predict(self, x, y):
        x_array, y_array = self.process_inputs(x, y)

        x_proj, y_proj = self.sess.run([self.x_proj, self.y_proj],
                                       feed_dict={self.x_place: x,
                                                  self.y_place: y})

        return x_proj, y_proj

    def __del__(self):
        self.sess.close()


def maxnorm_regularizer(threshold, axes=1, name='maxnorm', collection='maxnorm'):
    def maxnorm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None
    return maxnorm


def time_distributed_dense_layer(x, units, activation, name):
    """
    Layer that applies a dense matrix multiplication across
    all the samples as if they were one sample
    """
    input_shape = x.shape.as_list()
    x = tf.reshape(x, [-1, input_shape[2]])
    y = tf.layers.dense(x, units=units, activation=activation,
                        kernel_regularizer=maxnorm_regularizer(1.0),
                        name=name)
    y = tf.reshape(y, [1, -1, units])
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
    import numpy as np
    import tensorflow as tf
    from keras.datasets import mnist

    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    xx = xtrain[:100]
    yy = xtrain[100:200]
    
    cca_model = SparseCCA(nvecs=10, activation='linear')
    cca_model.fit(xx, yy, nb_epoch=5, batch_size=16)


    s="""
    x_array, y_array = cca_model.process_inputs(xx, yy)
    x_place = tf.placeholder(tf.float32, [1, None, x_array.shape[2]])
    y_place = tf.placeholder(tf.float32, [1, None, y_array.shape[2]])
    x_proj, y_proj = cca_model.inference(x_place, y_place)
    loss = cca_model.loss(x_proj, y_proj)
    train_op = cca_model.training(loss, optimizer='adam', learn_rate=1e-3)

    maxnorm_ops = tf.get_collection("maxnorm")

    nb_batches = int(np.ceil(x_array.shape[1] / float(16)))
    init_op = tf.global_variables_initializer()
    cca_model.sess.run(init_op)

    x_batch = x_array[:, :12, :]
    y_batch = y_array[:, :12, :]

    # run train step
    _loss, _ = cca_model.sess.run([loss, train_op],
                              feed_dict={
                                x_place: x_batch,
                                y_place: y_batch
                              })

    # check that weights have unit norm
    with tf.variable_scope('x_proj', reuse=True):
        xw = cca_model.sess.run(tf.get_variable('kernel'))
    with tf.variable_scope('y_proj', reuse=True):
        yw = cca_model.sess.run(tf.get_variable('kernel'))
    """    

