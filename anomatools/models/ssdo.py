# -*- coding: UTF-8 -*-
"""

Semi-Supervised Detection of Anomalies.

Reference:
    V. Vercruyssen, W. Meert, G. Verbruggen, K. Maes, R. Baumer, J. Davis.
    Semi-supervised anomaly detection with an application to water analytics.
    In IEEE International Conference on Data Mining, Singapore, 2018, pp. 527–536.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import copy
import numpy as np
import scipy.stats as sps

from collections import Counter
from sklearn.neighbors import BallTree
from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator

from .base import BaseDetector
from ..utils.copkmeans import COPKMeans
from ..utils.fastfuncs import fast_distance_matrix


# ----------------------------------------------------------------------------
# SSDO class
# ----------------------------------------------------------------------------

class SSDO(BaseEstimator, BaseDetector):
    """
    Parameters
    ----------
    k : int (default=30)
        Controls how many instances are updated by propagating the label of a
        single labeled instance.

    alpha : float (default=2.3)
        User influence parameter that controls the weight given to the
        unsupervised and label propragation components of an instance's
        anomaly score. Higher = more weight to supervised component.

    n_clusters : int (default=10)
        Number of clusters used for the COP k-means clustering algorithm.

    metric : string (default=euclidean)
        Distance metric for constructing the BallTree.

    unsupervised_prior : str (default='ssdo')
        Unsupervised prior:
            'ssdo'    --> SSDO baseline (based on constrained k-means clustering)
            'other'   --> use a different prior passed to SSDO

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
                 k=30,
                 alpha=2.3,
                 n_clusters=10,
                 unsupervised_prior='ssdo',
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
        self.nc = int(n_clusters)
        self.alpha = float(alpha)
        self.k = int(k)
        self.unsupervised_prior = str(unsupervised_prior).lower()

    def fit(self, X, y=None, prior=None):
        """ Fit the model on data X.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances. 
        y : np.array of shape (n_samples,), optional (default=None)
            The ground truth of the input instances.
        prior : np.array of shape (n_samples,), optional (default=None)
            Unsupervised prior of the input instances.

        Returns
        -------
        self : object
        """

        # check the inputs
        if y is None:
            y = np.zeros(len(X))
        X, y = check_X_y(X, y)

        # store label information
        ixl = np.where(y != 0)[0]
        self.feedback_ = y[ixl].copy()
        self.X_feedback_ = X[ixl, :].copy()

        # compute the prior
        if self.unsupervised_prior == 'ssdo':
            self._fit_prior_parameters(X, self.feedback_)
            prior = self._compute_prior(X)
        elif self.unsupervised_prior == 'other':
            if prior is None:
                raise ValueError('Prior cannot be None when `other` is selected')
        else:
            raise ValueError(self.unsupervised_prior,
                'is not in [ssdo, other]')
        self.prior_threshold_ = np.percentile(prior, 100*(1.0-self.c)) + self.tol

        # compute eta parameter
        self.eta_ = self._compute_eta(X)

        # feedback available
        if self.feedback_.any():
            self.scores_ = self._compute_posterior(X, prior, self.eta_)
        
        else:
            self.scores_ = prior.copy()
            
        self._process_anomaly_scores()

        return self
    
    def decision_function(self, X, prior=None):
        """ Compute the anomaly scores of X.
        
        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        prior : np.array of shape (n_samples,), optional (default=None)
            Unsupervised prior of the input instances.

        Returns
        -------
        scores : np.array of shape (n_samples,)
            The anomaly scores of the input instances.
        """

        # check the inputs
        X, _ = check_X_y(X, np.zeros(X.shape[0]))

        # compute the prior
        if self.unsupervised_prior == 'ssdo':
            prior = self._compute_prior(X)
        elif self.unsupervised_prior == 'other':
            if prior is None:
                raise ValueError('Prior cannot be None when `other` is selected')
        else:
            raise ValueError(self.unsupervised_prior,
                'is not in [ssdo, other]')

        # if no labels are available, reduce to unsupervised
        if not(self.feedback_.any()):
            return prior

        # compute posterior (includes squashing prior)
        posterior = self._compute_posterior(X, prior, self.eta_)

        return posterior
            
    def _compute_prior(self, X):
        """ Compute the constrained-clustering-based outlier score.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.

        Returns
        -------
        prior : np.array of shape (n_samples,)
            Unscaled unsupervised prior
        """

        n, _ = X.shape

        # predict the cluster labels + distances to the clusters
        _, labels, distances = self.clus.predict(X, include_distances=True)

        # compute the prior
        prior = np.zeros(n, dtype=float)
        for i, l in enumerate(labels):
            if self.max_intra_cluster[l] < self.tol:
                point_deviation = 1.0
            else:
                point_deviation = distances[i] / self.max_intra_cluster[l]
            prior[i] = (point_deviation * self.cluster_deviation[l]) / self.cluster_sizes[l]

        return prior

    def _compute_posterior(self, X, prior, eta):
        """ Update the prior score with label propagation.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        prior : np.array of shape (n_samples,), optional (default=None)
            Unsupervised prior of the input instances.
        eta : float
            Eta parameter is the harmonic mean of the k-distances.

        Returns
        -------
        posterior : np.array of shape (n_samples)
            Posterior anomaly score between 0 and 1.
        """

        n, _ = X.shape

        # squash the prior
        prior = self._squashing_function(prior, self.prior_threshold_)

        # labeled examples
        ixa = np.where(self.feedback_ == 1.0)[0]
        ixn = np.where(self.feedback_ == -1.0)[0]

        # compute limited distance matrices (to normals, to anomalies)
        Dnorm = fast_distance_matrix(X, self.X_feedback_[ixn, :])
        Danom = fast_distance_matrix(X, self.X_feedback_[ixa, :])

        # compute posterior
        posterior = np.zeros(n, dtype=float)
        for i in range(n):
            # weighted distance to anomalies & normals
            da = np.sum(self._ssdo_squashing_function(Danom[i, :], eta))
            dn = np.sum(self._ssdo_squashing_function(Dnorm[i, :], eta))
            # posterior
            z = 1.0 / (1.0 + self.alpha * (da + dn))
            posterior[i] = z * (prior[i] + self.alpha * da)

        return posterior

    def _fit_prior_parameters(self, X, y):
        """ Fit the parameters for computing the prior score:
        
        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        prior : np.array of shape (n_samples,), optional (default=None)
            Unsupervised prior of the input instances.

        Returns
        -------
        self : object
        """

        # construct cannot-link constraints + remove impossible cannot-links
        ixn = np.where(y == -1.0)[0]
        ixa = np.where(y == 1.0)[0]
        cl = np.array(np.meshgrid(ixa, ixn)).T.reshape(-1,2)

        # cluster
        self.clus = COPKMeans(n_clusters=self.nc)
        centroids, labels = self.clus.fit_predict(X, cannot_link=cl)
        self.nc = self.clus.n_clusters

        # cluster sizes (Counter sorts by key!)
        self.cluster_sizes = np.array(list(Counter(labels).values())) / max(Counter(labels).values())

        # compute the max intra-cluster distance
        self.max_intra_cluster = np.zeros(self.nc, dtype=np.float)
        for i, l in enumerate(labels):
            c = centroids[l, :]
            d = np.linalg.norm(X[i, :] - c)
            if d > self.max_intra_cluster[l]:
                self.max_intra_cluster[l] = d

        # compute the inter-cluster distances
        if self.nc == 1:
            self.cluster_deviation = np.array([1])
        else:
            inter_cluster = np.ones(self.nc, dtype=np.float) * np.inf
            for i in range(self.nc):
                for j in range(self.nc):
                    if i != j:
                        d = np.linalg.norm(centroids[i, :] - centroids[j, :])
                        if not(d < self.tol) and d < inter_cluster[i]:
                            inter_cluster[i] = d
        self.cluster_deviation = inter_cluster / max(inter_cluster)

        return self

    def _compute_eta(self, X):
        """ Compute the eta parameter.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        
        Returns
        -------
        eta : float
            Eta parameter is the harmonic mean of the k-distances.
        """

        n, _ = X.shape

        # construct KD-tree
        tree = BallTree(X, leaf_size=32, metric=self.metric)
        D, _ = tree.query(X, k=self.k+1, dualtree=True)
        d = D[:, -1].flatten()

        # compute eta as the harmonic mean of the k-distances
        filler = min(d[d > 0.0])
        d[d == 0.0] = filler
        eta = sps.hmean(d)

        if eta < self.tol:
            eta = self.tol

        return eta

    def _ssdo_squashing_function(self, x, gamma):
        """ Compute the value of x under squashing function.
        """
        
        return np.exp(np.log(0.5) * np.power(x / gamma, 2))
