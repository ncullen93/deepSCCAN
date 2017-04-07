

from __future__ import division

from keras import backend as K
from keras.constraints import Constraint
import theano
import theano.tensor
import theano.tensor.nlinalg as TN
import theano.tensor.basic as TB


def batch_cca_loss(y_true,y_pred):
    """
    Return Sum of the diagonal - Sum of upper and lower triangles
    """
    trace       = TN.trace(y_true[0,:,:])
    triu_sum    = K.sum(K.abs(TB.triu(y_true[0,:,:],k=1)))
    tril_sum    = K.sum(K.abs(TB.tril(y_true[0,:,:],k=-1)))
    return trace - tril_sum - triu_sum


def cca_loss(y_true,y_pred):
    return -1*y_pred


class UnitNormWithNonnneg(Constraint):
    """
    - Nonnegativity constraint
    - Max L1 Norm constraint
    """
    def __init__(self, nonneg=False, axis=0):
        self.nonneg = nonneg
        self.axis = axis

    def __call__(self, p):
        p = p / (K.epsilon() + K.sqrt(K.sum(K.square(p), axis=self.axis, keepdims=True)))
        if self.nonneg:
            p *= K.cast(p >= 0., K.floatx())

        return p

    def get_config(self):
        return {'name': self.__class__.__name__,
                'nonneg': self.nonneg,
                'axis': self.axis}


class PickMax(Constraint):
    """
    Only keeps max value from each component. So each component would have 1 non-zero value.
    """
    def __init__(self,axis=-1):
        self.nonneg = nonneg
        self.axis = axis

    def __call__(self, p):
        p *= K.cast(K.equal(p,K.max(p)), K.floatx())
        return p

    def get_config(self):
        return {'name': self.__class__.__name__,
            'axis': self.axis}

