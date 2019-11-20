# -*- coding: UTF-8 -*-
"""

k-nearest neighbor anomaly detection.

Reference:
    S. Ramaswamy, R. Rastogi, and K. Shim. Efficient algorithms for mining outliers from large data sets.
    In Proceedings of the 2000 ACM SIGMOD international conference on Management of data, vol. 29, no. 2. ACM, 2000, pp. 427–438.

:author: Vincent Vercruyssen
:year: 2018
:license: Apache License, Version 2.0, see LICENSE for details.

"""

import math
import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------------------------------------------
# KNNO class
# ----------------------------------------------------------------------------

class KNNO(BaseEstimator):
    """
    Parameters
    ----------
    k : int (default=10)
        Number of nearest neighbors.

    contamination : float (default=0.1)
        Estimate of the expected percentage of anomalies in the data.

    metric : string (default=euclidean)
        Distance metric for constructing the BallTree: euclidean, cosine

    Comments
    --------
    - The parameter k should be carefully chosen depending on whether you want
    to predict on a separate test set or not.
    - The number of neighbors cannot be larger than the number of instances in
    the data: automatically correct if necessary.
    """

    def __init__(self,
                 k=10,                      # the number of neighbors
                 contamination=0.1,         # expected proportion of anomalies in the data
                 metric='euclidean',        # distance metric used
                 tol=1e-8,                  # tolerance
                 verbose=False):
        super().__init__()

        # instantiate parameters
        self.k = int(k)
        self.contamination = float(contamination)
        if metric not in BallTree.valid_metrics + ['cosine']:
            raise BaseException('Invalid distance metric!')
        self.metric = str(metric)
        self.tol = float(tol)
        self.verbose = bool(verbose)

    def fit_predict(self, X, y=None):
        """ Fit the model to the training set X and returns the anomaly score
            of the instances in X.

        :param X : np.array(), shape (n_samples, n_features)
            The samples to compute anomaly score w.r.t. the training samples.
        :param y : np.array(), shape (n_samples), default = None
            Labels for examples in X.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        return self.fit(X, y).predict(X)

    def fit(self, X, y=None):
        """ Fit the model using data in X.

        :param X : np.array(), shape (n_samples, n_features)
            The samples to compute anomaly score w.r.t. the training samples.
        :param y : np.array(), shape (n_samples), default = None
            Labels for examples in X.

        :returns self : object
        """

        # check input
        if y is None:
            y = np.zeros(len(X))
        X, y = check_X_y(X, y)
        
        n, _ = X.shape

        self.nn = self._check_valid_number_of_neighbors(n)
        if self.metric in BallTree.valid_metrics:
            self.tree = BallTree(X, leaf_size=32, metric=self.metric)
        elif self.metric == 'cosine':
            def angular_distance_metric(a, b):
                # in order to be used with BallTree, must be a true distance metric!
                # satisfy: (1) non-negativity, (2) identify, (3) symmetry, (4) triangle inequality
                cs = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).flatten()[-1]
                if abs(cs - 1.0) < self.tol:
                    cs = 1.0
                return float(math.acos(cs) / math.pi)
            self.tree = BallTree(X, leaf_size=32, metric='pyfunc', func=angular_distance_metric)

        return self

    def predict(self, X):
        """ Compute the anomaly score + predict the label of instances in X.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        # check input
        X, _ = check_X_y(X, np.zeros(len(X)))

        n, _ = X.shape

        # compute the outlier score
        knn_score = np.zeros(n, dtype=float)
        for i, x in enumerate(X):
            dist, _ = self.tree.query([x], k=max(2, self.nn))
            knn_score[i] = dist.flatten()[-1]

        # normalize the score between 0 and 1
        y_score = (knn_score - min(knn_score)) / (max(knn_score) - min(knn_score))

        # prediction threshold + absolute predictions
        self.threshold = np.sort(y_score)[int(n * (1.0 - self.contamination))]
        y_pred = np.ones(n, dtype=float)
        y_pred[y_score < self.threshold] = -1

        return y_score, y_pred

    def _check_valid_number_of_neighbors(self, n_samples):
        """ Check if the number of nearest neighbors is valid and correct.

        :param n_samples : int
            Number of samples in the data.
        """

        return min(n_samples, self.k)
