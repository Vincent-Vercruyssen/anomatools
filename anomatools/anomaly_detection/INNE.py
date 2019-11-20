# -*- coding: UTF-8 -*-
"""

Isolation nearest neighbor ensembles.

Reference:
    T. R. Bandaragoda, K. Ming Ting, D. Albrecht, F. T. Liu, Y. Zhu, and J. R. Wells.
    Isolation-based anomaly detection using nearest-neighbor ensembles.
    In Computational Intelligence, vol. 34, 2018, pp. 968-998.

:author: Vincent Vercruyssen
:year: 2018
:license: Apache License, Version 2.0, see LICENSE for details.

"""

import math
import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import BallTree



# ----------------------------------------------------------------------------
# INNE class
# ----------------------------------------------------------------------------

class INNE(BaseEstimator):
    """
    Parameters
    ----------
    t : int (default=100)
        Number of estimators to construct.

    n : int (default=16)
        Number of examples in each subsample.

    contamination : float (default=0.1)
        Estimate of the expected percentage of anomalies in the data.

    metric : string (default=euclidean)
        Distance metric for constructing the BallTree: euclidean, cosine

    Comments
    --------
    - The number of examples in each subsample cannot be larger than the number of instances in
    the data: automatically correct if necessary.
    """

    def __init__(self,
                 t=100,                     # number of ensemble members
                 n=16,                      # sample for each ensemble member
                 contamination=0.1,         # expected proportion of anomalies in the data
                 metric='euclidean',        # distance metric to use
                 tol=1e-8,                  # tolerance
                 verbose=False):
        super().__init__()

        # instantiate the parameters
        self.t = int(t)
        self.n = int(n)
        self.contamination = float(contamination)
        if metric == 'cosine':
            self.metric = cosine_similarity
        else:
            try:
                self.metric = DistanceMetric.get_metric(metric)
            except ValueError as e:
                raise BaseException(e)
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

        # set value for n
        self.n = min(self.n, n)

        # construct the ensembles with random sampling of the points
        self._ensemble = []
        for _ in range(self.t):
            # random sample
            ixs = np.random.choice(n, self.n, replace=False)
            
            sphere = hyperSphere(X[ixs, :])
            self._ensemble.append(sphere)
        
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

        # compute the anomaly score using each member of the ensemble
        iscores = np.zeros(n, dtype=float)
        for sphere in self._ensemble:
            score = sphere.compute_isolation_score(sphere, X)
            iscores = iscores + score
        iscores = iscores / self.t

        # prediction labels
        self.threshold = np.sort(iscores)[int(n * (1.0 - self.contamination))]
        y_pred = np.ones(n, dtype=float)
        y_pred[iscores < self.threshold] = -1

        return iscores, y_pred


# ----------------------------------------------------------------------------
# single member of the INNE ensemble
# ----------------------------------------------------------------------------

class hyperSphere:
    
    def __init__(self, X):
        # constructs hypersphere
        self.nm = X.shape[0]
        self.nn_tree = BallTree(X, leaf_size=16, metric='euclidean')
        nn_dists, nn_ixs = self.nn_tree.query(X, k=2)
        
        # radii
        eps = 1e-10
        self.radii = nn_dists[:, 1].flatten() + eps
        
        # isolation scores
        self.scores = 1.0 - (self.radii[nn_ixs[:, 1].flatten()] / self.radii)
    
    def compute_isolation_score(self, sphere, X):
        # compute isolation score for sample X
        n, _ = X.shape
        scores = np.ones(n, dtype=np.float)
        
        s_dists, s_ixs = sphere.nn_tree.query(X, k=self.nm)
        
        for i in range(n):
            cr = self.radii[s_ixs[i, :].flatten()]
            
            # belongs to these spheres
            ix_m = np.where(s_dists[i, :].flatten() <= cr)[0]
            
            # does not belong to sphere
            if len(ix_m) == 0:
                continue
            
            # sphere with smallest radius
            ixs = np.argmin(cr[ix_m])
            ns = s_ixs[i, ixs]
            scores[i] = self.scores[ns]
            
        return scores
