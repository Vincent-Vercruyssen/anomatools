""" Semi-Supervised Detection of Anomalies """

# Authors: Vincent Vercruyssen

import numpy as np
import scipy.stats as sps

from collections import Counter
from scipy.spatial import cKDTree

from ..clustering.COPKMeans import COPKMeans
from ..utils.fastfuncs import fast_distance_matrix
from ..utils.validation import check_X_y


class SSDO:
    """ Semi-Supervised Detection of Anomalies (SSDO)

    Parameters
    ----------
    n_clusters : int (default=10)
        Number of clusters used for the COP k-means clustering algorithm.

    alpha : float (default=2.3)
        User influence parameter that controls the weight given to the
        unsupervised and label propragation components of an instance's
        anomaly score.

    k : int (default=30)
        Controls how many instances are updated by propagating the label of a
        single labeled instance.

    contamination : float (default=0.1)
        Estimate of the expected percentage of anomalies in the data.
    """

    def __init__(self, n_clusters=10, alpha=2.3, k=30, contamination=0.1,
                 tol=1e-8, verbose=False):

        self.nc = n_clusters
        self.alpha = alpha
        self.k = k
        self.c = contamination
        self.tol = tol
        self.verbose = verbose

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

        return self.fit(X, y)._predict()

    def fit(self, X, y=None):
        """ Fit the model using data in X.

        :param X : np.array(), shape (n_samples, n_features)
            The samples to compute anomaly score w.r.t. the training samples.
        :param y : np.array(), shape (n_samples), default = None
            Labels for examples in X.

        :returns self : object
        """

        X, y = check_X_y(X, y)

        n, _ = X.shape
        if y is None:
            y = np.zeros(n, dtype=int)

        # compute the constrained-clustering-based score (scaled)
        prior = self._compute_prior(X, y)

        # compute eta parameter
        eta = self._compute_eta(X)

        # update the clustering score with label propragation
        self.posterior = self._compute_posterior(X, y, prior, eta)

        return self

    def _predict(self):
        """ Compute the anomaly score + predict the label of instances in X.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        n = len(self.posterior)

        # compute y_score
        y_score = self.posterior

        # compute y_pred
        offset = np.sort(y_score)[int(n * (1.0 - self.c))]
        y_pred = np.ones(n, dtype=int)
        y_pred[y_score < offset] = -1

        return y_score, y_pred

    def _compute_prior(self, X, y):
        """ Compute the constrained-clustering-based outlier score.

        :returns prior : np.array(), shape (n_samples)
            Prior anomaly score between 0 and 1.
        """

        n, _ = X.shape

        # constrained clustering of the data
        labels, centroids = self._constrained_clustering(X, y)

        # compute cluster sizes
        cluster_size = np.array(list(Counter(labels).values())) / max(Counter(labels).values())

        # compute the max intra-cluster distance + distances to centroids
        distances = np.zeros(n)
        max_intra_cluster = np.zeros(self.nc)
        for i, l in enumerate(labels):
            c = centroids[l, :]
            d = np.linalg.norm(X[i, :] - c)
            if d > max_intra_cluster[l]:
                max_intra_cluster[l] = d
            distances[i] = d

        # compute the inter-cluster distance
        if self.nc == 1:
            inter_cluster = np.ones(self.nc)
        else:
            inter_cluster = np.ones(self.nc) * np.inf
            for i in range(self.nc):
                for j in range(self.nc):
                    if i != j:
                        d = np.linalg.norm(centroids[i, :] - centroids[j, :])
                        if not(d < self.tol) and d < inter_cluster[i]:
                            inter_cluster[i] = d
        cluster_deviation = inter_cluster / max(inter_cluster)

        # compute the prior
        prior = np.zeros(n)
        for i, l in enumerate(labels):
            if max_intra_cluster[l] < self.tol:
                point_deviation = 1.0
            else:
                point_deviation = distances[i] / max_intra_cluster[l]
            prior[i] = (point_deviation * cluster_deviation[l]) / cluster_size[l]

        # scale the prior using the squashing function
        gamma = np.sort(prior)[int(n * (1.0 - self.c))] + self.tol
        prior = np.array([1 - self._squashing_function(x, gamma) for x in prior])

        return prior

    def _compute_posterior(self, X, y, prior, eta):
        """ Update the clustering score with label propagation.

        :returns posterior : np.array(), shape (n_samples)
            Posterior anomaly score between 0 and 1.
        """

        n, _ = X.shape
        ix_a = np.where(y == 1)[0]
        ix_n = np.where(y == -1)[0]
        if len(ix_a) + len(ix_n) > 0:
            labels = True
        else:
            labels = False

        # compute limited distance matrices (to normals, to anomalies)
        if labels:
            Dnorm = fast_distance_matrix(X, X[ix_n, :])
            Danom = fast_distance_matrix(X, X[ix_a, :])

        # compute posterior
        if labels:
            posterior = np.zeros(n)
            for i in range(n):
                # weighted distance to anomalies & normals
                da = np.sum(self._squashing_function(Danom[i, :], eta))
                dn = np.sum(self._squashing_function(Dnorm[i, :], eta))
                # posterior
                z = 1.0 / (1.0 + self.alpha * (da + dn))
                posterior[i] = z * (prior[i] + self.alpha * da)
        else:
            posterior = prior

        return posterior

    def _constrained_clustering(self, X, y):
        """ Constrained clustering of X with labels y.

        :returns labels : np.array(), shape (n_samples)
            Cluster labels of the instances in X [0 ... nc-1].
        :returns centroids : np.array(), shape (n_clusters, n_features)
            Cluster centroids.
        """

        # construct cannot-link constraints + remove impossible cannot-links
        ix_n = np.where(y == -1)[0]
        ix_a = np.where(y == 1)[0]
        cl = np.array(np.meshgrid(ix_a, ix_n)).T.reshape(-1,2)

        # cluster
        clus = COPKMeans(n_clusters=self.nc)
        clus.fit(X, cannot_link=cl)

        # update self.nc
        self.nc = clus.n_clusters
        labels, centroids = clus.labels_, clus.cluster_centers_

        return labels, centroids

    def _compute_eta(self, X):
        """ Compute the eta parameter.

        :returns eta : float
            Eta parameter is the harmonic mean of the k-distances.
        """

        n, _ = X.shape

        # construct KD-tree
        tree = cKDTree(X, leafsize=16)

        # query distance to k'th nearest neighbor of each point
        d = np.zeros(n)
        for i, x in enumerate(X):
            dist, _ = tree.query(x, k=self.k+1)
            d[i] = dist[-1]

        # compute eta as the harmonic mean of the k-distances
        filler = min(d[d > 0.0])
        d[d == 0.0] = filler
        eta = sps.hmean(d)

        if eta < self.tol:
            eta = self.tol

        return eta

    def _squashing_function(self, x, p):
        """ Compute the value of x under squashing function with parameter p. """
        return np.exp(np.log(0.5) * np.power(x / p, 2))
