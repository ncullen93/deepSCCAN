"""
Hyperparameter Optimization routines for CCA models
"""

def mean_cca_score(estimator, X, y, sample_weight=None):
    return np.mean(np.abs(estimator.evaluate(X, y)))

def sum_cca_score(estimator, X, y, sample_weight=None):
    return np.sum(np.abs(estimator.evaluate(X, y)))

if __name__ == '__main__':
    import numpy as np
    from sparse_cca import SparseCCA
    from sklearn.model_selection import GridSearchCV
    from keras.datasets import mnist
    (x, _), _ = mnist.load_data()
    x = (x - np.mean(x)) / np.std(x)
    x_train = x[:1000,:,:14]
    y_train = x[:1000,:,14:]

    param_grid = [
        {
            'sparsity': [0.1, 0.2]    
        }
    ]
    fit_params = {
        'nb_epoch' : 10
    }

    cca_model = SparseCCA(ncomponents=5, nb_epoch=50, deflation=False,
                          learn_rate=1e-4, verbose=False)
    #cca_model.fit(x_train, y_train)
    from sklearn.model_selection import PredefinedSplit, ShuffleSplit

    # -1 means always in train set, 0,1,2,etc means new test set
    #cv_split = PredefinedSplit(np.hstack((-1*np.ones(600), np.zeros(400))))
    cv_split = ShuffleSplit(n_splits=1, test_size=0.20, random_state=0)
    clf = GridSearchCV(cca_model, param_grid=param_grid, 
        cv=cv_split, fit_params=fit_params, n_jobs=1, verbose=10)

    clf.fit(x_train, y_train)



