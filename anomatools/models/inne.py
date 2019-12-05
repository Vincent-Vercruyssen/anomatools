# -*- coding: UTF-8 -*-
"""

Isolation nearest neighbor ensembles.

Reference:
    T. R. Bandaragoda, K. Ming Ting, D. Albrecht, F. T. Liu, Y. Zhu, and J. R. Wells.
    Isolation-based anomaly detection using nearest-neighbor ensembles.
    In Computational Intelligence, vol. 34, 2018, pp. 968-998.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import math
import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator
from sklearn.neighbors import BallTree

from .base import BaseDetector


# ----------------------------------------------------------------------------
# iNNE class
# ----------------------------------------------------------------------------

class iNNE(BaseEstimator, BaseDetector):
    """ iNNE class for anomaly/outlier detection.

    Parameters
    ----------
    n_members : int (default=100)
        Number of estimators to construct.

    sample_size : int (default=16)
        Number of examples in each subsample.

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
                 n_members=100,
                 sample_size=16,
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
        self.n_members = int(n_members)
        self.sample_size = int(sample_size)

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

        # correct sample size
        n = X.shape[0]
        self.sample_size = min(self.sample_size, n)

        # construct the ensemble
        self.ensemble_ = []
        for _ in range(self.n_members):
            # random sample
            ixs = np.random.choice(n, self.sample_size, replace=False)
            
            sphere = HyperSphere(X[ixs, :], self.metric)
            self.ensemble_.append(sphere)

        # COST: anomaly scores of the training data
        self.scores_ = self.decision_function(X)
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
        scores = np.zeros(X.shape[0], dtype=float)
        for sphere in self.ensemble_:
            score = sphere.compute_isolation_score(sphere, X)
            scores = scores + score
        scores = scores / self.n_members

        return scores


# ----------------------------------------------------------------------------
# single member of the iNNE ensemble
# ----------------------------------------------------------------------------

class HyperSphere:
    
    def __init__(self, X, metric):
        # constructs hypersphere
        self.nm = X.shape[0]
        self.nn_tree = BallTree(X, leaf_size=16, metric=metric)
        nn_dists, nn_ixs = self.nn_tree.query(X, k=2)
        
        # radii
        self.radii = nn_dists[:, 1].flatten() + 1e-10
        
        # isolation scores
        self.scores = 1.0 - (self.radii[nn_ixs[:, 1].flatten()] / self.radii)
    
    def compute_isolation_score(self, sphere, X):
        # compute isolation score for sample X
        n, _ = X.shape
        scores = np.ones(n, dtype=float)
        
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
