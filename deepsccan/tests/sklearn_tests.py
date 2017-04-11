"""
Tests for sklearn compatibility of estimators
"""

from sklearn.utils.estimator_checks import check_estimator


def check_SparseCCA():
    from ..tf_models.sparse_cca import SparseCCA
    check_estimator(SparseCCA)

if __name__ == '__main__':
    check_SparseCCA()