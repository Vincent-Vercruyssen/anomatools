# -*- coding: UTF-8 -*-
"""

k-nearest neighbor anomaly detection.

Reference:
    S. Ramaswamy, R. Rastogi, and K. Shim. Efficient algorithms for mining outliers from large data sets.
    In Proceedings of the 2000 ACM SIGMOD international conference on Management of data, vol. 29, no. 2. ACM, 2000, pp. 427–438.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import math
import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import BallTree

from .base import BaseDetector


# ----------------------------------------------------------------------------
# kNNO class
# ----------------------------------------------------------------------------

class kNNO(BaseEstimator, BaseDetector):
    """ k Nearest neighbor outlier detection.

    Parameters
    ----------
    k : int (default=10)
        Number of nearest neighbors.

    weighted : bool, optional (default=False)
        Weight the scores using distance.

    metric : string (default=euclidean)
        Distance metric for constructing the BallTree.
    
    Attributes
    ----------
    scores_ : np.array of shape (n_samples,)
        The anomaly scores of the training data (higher = more abnormal).
    
    threshold_ : float
        The cutoff threshold on the anomaly score separating the normals
        from the anomalies. This is based on the `contamination` parameter.

    labels_ : np.array of shape (n_samples,)
        Binary anomaly labels (-1 = normal, +1 = anomaly).
    """

    def __init__(self,
                 k=10,
                 weighted=False,
                 contamination=0.1,
                 metric='euclidean',
                 tol=1e-8,
                 verbose=False):
        super().__init__(
            contamination=contamination,
            metric=metric,
            tol=tol,
            verbose=verbose)

        # instantiate the parameters
        self.k = int(k)
        self.weighted = bool(weighted)

    def fit(self, X, y=None):
        """ Fit the model on data X.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
        """

        # check the inputs
        X, y = check_X_y(X, np.zeros(X.shape[0]))

        # correct number of neighbors
        n = X.shape[0]
        self.k = min(self.k, n)

        # construct the neighbors tree
        self.tree_ = BallTree(X, leaf_size=32, metric=self.metric)

        # COST: anomaly scores of the training data
        D, _ = self.tree_.query(X, k=self.k+1, dualtree=True)
        self.scores_ = self._get_distances_by_method(D)
        self._process_anomaly_scores()
        
        return self

    def decision_function(self, X):
        """ Compute the anomaly scores of X.
        
        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.

        Returns
        -------
        scores : np.array of shape (n_samples,)
            The anomaly scores of the input instances.
        """
        
        # check the inputs
        X, _ = check_X_y(X, np.zeros(X.shape[0]))

        # compute the anomaly scores
        D, _ = self.tree_.query(X, k=self.k, dualtree=True)
        scores = self._get_distances_by_method(D)

        return scores

    def _get_distances_by_method(self, D):
        """ Determine how to process the neighbor distances.

        Parameters
        ----------
        D : np.array of shape (n_samples, k)
            Distance matrix.

        Returns
        -------
        dist : np.array of shape (n_samples,)
            The distance outlier scores.
        """

        if self.weighted:
            return np.mean(D[:, 1:], axis=1)
        else:
            return D[:, -1].flatten()
