# -*- coding: UTF-8 -*-
"""

Functions for fast distance computations.

:author: Vincent Vercruyssen
:year: 2018
:license: Apache License, Version 2.0, see LICENSE for details.

"""

import numpy as np


# ----------------------------------------------------------------------------
# functions
# ----------------------------------------------------------------------------

def fast_distance_matrix(X, Y):
    """ Compute distance matrix between instances in sets X and Y

    :param X : np.array(), shape (n_samples1, n_features)
        First set of samples.
    :param Y : np.array(), shape (n_samples2, n_features)
        Second set of samples.

    :returns D : np.array(), shape (n_samples1, n_samples2)
        Euclidean distance between any pair of instances in X and Y.
    """

    if len(X.shape) == 1:
        X = np.reshape(X, (1, -1))
    if len(Y.shape) == 1:
        Y = np.reshape(Y, (1, -1))
    n, _ = X.shape
    m, _ = Y.shape
    dx = np.sum(X ** 2, axis=1)
    dy = np.sum(Y ** 2, axis=1)
    H1 = np.tile(dx, (m, 1))
    H2 = np.tile(dy, (n, 1))
    H3 = np.dot(X, Y.T)
    D = H1.T + H2 - 2 * H3
    D[D < 0.0] = 0.0  # issues with small numbers
    return np.sqrt(D)
