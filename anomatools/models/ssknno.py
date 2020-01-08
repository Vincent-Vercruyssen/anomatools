# -*- coding: UTF-8 -*-
"""

Semi-supervised k-nearest neighbor outlier detection.

Reference:
    V. Vercruyssen, W. Meert, J. Davis.
    Transfer Learning for Anomaly Detection through Localized and Unsupervised Instance Selection.
    In AAAI Conference on Artificial Intelligence, New York, 2020.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator
from sklearn.neighbors import BallTree

from .base import BaseDetector


# ----------------------------------------------------------------------------
# SSkNNO class
# ----------------------------------------------------------------------------

class SSkNNO(BaseEstimator, BaseDetector):
    """ Semi-supervised k-nearest neighbors anomaly detection.

    Parameters
    ----------
    k : int (default=10)
        Number of nearest neighbors.

    weighted : bool, optional (default=False)
        Weight the scores using distance.

    metric : string (default=euclidean)
        Distance metric for constructing the BallTree.

    supervision : str (default=loose)
        How to compute the supervised score component.
        'loose'     --> use all labeled instances in the set of nearest neighbors
        'strict'    --> use only instances that also count the instance among their neighbors
   
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
                 supervision='loose',
                 contamination=0.1,
                 metric='euclidean',
                 tol=1e-8,
                 verbose=False):
        super().__init__(
            contamination=contamination,
            metric=metric,
            tol=tol,
            verbose=verbose)

        # initialize parameters
        self.k = int(k)
        self.weighted = bool(weighted)
        self.supervision = str(supervision)
        
    def fit(self, X, y=None):
        """ Fit the model on data X.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances. 
        y : np.array of shape (n_samples,), optional (default=None)
            The ground truth of the input instances.

        Returns
        -------
        self : object
        """

        # check the inputs
        if y is None:
            y = np.zeros(len(X))
        X, y = check_X_y(X, y)
        self.feedback_ = y.copy()

        # correct number of neighbors
        n = X.shape[0]
        self.k = min(self.k, n)

        # construct tree
        self.tree_ = BallTree(X, leaf_size=32, metric=self.metric)

        # unsupervised score + threshold
        D, _ = self.tree_.query(X, k=self.k+1, dualtree=True)
        prior = self._compute_prior(D)
        self.prior_threshold_ = np.percentile(prior, 100*(1.0-self.c)) + self.tol
        
        # feedback available
        if self.feedback_.any():
            # collect ALL the nearest neighbors in the radius
            self.radii_ = D[:, -1].flatten() + self.tol
            Ixs_radius, D_radius = self.tree_.query_radius(
                X, r=self.radii_, return_distance=True, count_only=False)
            
            # compute posterior (includes squashing prior)
            self.scores_ = self._compute_posterior(Ixs_radius, D_radius, prior)

        else:
            self.scores_ = prior.copy()
        
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

        # unsupervised scores
        D, _ = self.tree_.query(X, k=self.k, dualtree=True)
        prior = self._compute_prior(D)

        # if no labels are available, reduce to kNNO
        if not(self.feedback_.any()):
            return prior

        # collect ALL the nearest neighbors in the radius
        nn_radii = D[:, -1].flatten() + self.tol
        Ixs_radius, D_radius = self.tree_.query_radius(
            X, r=nn_radii, return_distance=True, count_only=False)

        # compute posterior (includes squashing prior)
        posterior = self._compute_posterior(Ixs_radius, D_radius, prior)

        return posterior

    def _compute_prior(self, D):
        """ Compute the supervised scores of the points.
        """

        return self._get_distances_by_method(D)

    def _compute_posterior(self, Ixs, D, prior):
        """ Compute the kNNO scores of the points.
        """

        n = len(prior)

        # squash the prior
        prior = self._squashing_function(prior, self.prior_threshold_)
        
        # compute the posterior
        posterior = np.zeros(n, dtype=float)
        for i in range(n):
            # k nearest neighbors and indices
            ndists = D[i]  # could be longer than k
            nixs = Ixs[i]
            nn = len(ndists)  # nn can now change per point

            # labels of the neighbors, weights
            labels = self.feedback_[nixs]
            w = np.power(1.0 / (ndists + self.tol), 2)

            # supervised score component
            ixl = np.where(labels != 0.0)[0]
            if len(ixl) > 0:
                # reverse nearest neighbors
                reverse_nn = np.where(ndists <= self.radii_[nixs])[0]
                reverse_nn = np.intersect1d(ixl, reverse_nn)
                Ws = len(reverse_nn) / nn
                # previous: Ws = np.sum(w[ixl]) / np.sum(w)

                # supervised score
                if self.supervision == 'loose':
                    ixa = np.where(labels > 0)[0]
                    Ss = np.sum(w[ixa]) / np.sum(w[ixl])
                    # previous: Ss = len(ixa) / len(ixl)
                elif self.supervision == 'strict':
                    if len(reverse_nn) > 0:
                        ixa = np.where(labels[reverse_nn] > 0)[0]
                        Ss = np.sum(w[ixa]) / np.sum(w[reverse_nn])
                    else:
                        Ss = 0.0
                else:
                    raise ValueError(self.supervision,
                        'is not in [loose, strict]')
            
            # supervised plays no role
            else:
                Ss = 0.0
                Ws = 0.0

            # combine supervised and unsupervised
            posterior[i] = (1.0 - Ws) * prior[i] + Ws * Ss

        return posterior

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
