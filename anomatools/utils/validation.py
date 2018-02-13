""" Functions for validation """

# Authors: Vincent Vercruyssen

import warnings
import numpy as np


def check_X_y(X, y):
    """ Check whether X is array.

    :param X : np.array(), shape (n_samples, n_features)
        Input to be checked.
    :param y : np.array(), shape (n_samples)
        Input to be checked.

    :returns X_converted : np.array()
        Converted and validated X.
    :returns y_converted : np.array()
        Converted and validated y.
    """

    # check if arrays
    if not(isinstance(X, np.ndarray)):
        raise ValueError('Input X is not a numpy array.')

    if not(isinstance(y, np.ndarray)):
        if y is not None:
            raise ValueError('Input y is not a numpy array or none.')

    # check if X has required dimensions
    if len(X.shape) == 1:
        msg = ('X contains only a single example.')
        warnings.warn(msg)
        X = np.reshape(X, (1, -1))

    return X, y
