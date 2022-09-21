"""
Cluster-based anomaly detection (SSDO paper).

Reference:
    V. Vercruyssen, W. Meert, G. Verbruggen, K. Maes, R. Baumer, J. Davis.
    Semi-supervised anomaly detection with an application to water analytics.
    In IEEE International Conference on Data Mining, Singapore, 2018, pp. 527–536.

:author: Vincent Vercruyssen (2022)
:license: Apache License, Version 2.0, see LICENSE for details.
"""


import math
import numpy as np

from collections import Counter
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans

from ._base import BaseDetector
from .scalers import SquashScaler


# ----------------------------------------------------------------------------
# kNNo detector
# ----------------------------------------------------------------------------


class ClusterBasedAD(BaseEstimator, BaseDetector):
    """
    The cluster-based anomaly detector finds anomalies by
    computing three measures related to how each example
    clusters in a given dataset.

    Parameters
    ----------
    base_clusterer : object, optional
        Base cluster method. Only works with objects that have a
        'predict()' method returning the cluster labels.

    scaler : object, optional
        Scaling applied to the anomaly scores.

    metric : str or callable, optional
        The distance metric. Currently, only sklearn metrics.

    metric_params : dict, optional
        Additional keyword arguments for the metric function.

    verbose : bool, optional
        Verbosity.
    """

    def __init__(
        self,
        base_clusterer=KMeans(),
        scaler=SquashScaler(),
        metric="euclidean",
        metric_params={},
        verbose=False,
    ):
        super().__init__(
            scaler=scaler,
            metric=metric,
            metric_params=metric_params,
            verbose=verbose,
        )

        self.base_clusterer = base_clusterer

    # ----------------------------- OVERRIDE -----------------------------
    def _train(self, X, y=None, sample_weight=None):
        """Fit the detector.

        Parameters
        ----------
        X : np.ndarray
            The training instances.

        y : np.ndarray, optional
            The training instance class labels as ints.

        sample_weight : np.ndarray, optional
            The sample weights of the training instances.
        """

        # construct the cannot-link constraints
        # + remove impossible cannot-links
        ixn = np.where(y == -1.0)[0]
        ixa = np.where(y == 1.0)[0]
        cl = np.array(np.meshgrid(ixa, ixn)).T.reshape(-1, 2)

        # cluster and predict the labels
        # TODO: add support for COPKMeans again
        self.base_clusterer.fit(X)
        centroids = self.base_clusterer.cluster_centers_
        labels = self.base_clusterer.labels_
        self.nc = self.base_clusterer.n_clusters

        # cluster sizes (Counter sorts by key!)
        self.cluster_sizes = np.array(list(Counter(labels).values())) / max(
            Counter(labels).values()
        )

        # compute the max intra-cluster distance
        self.max_intra_cluster = np.zeros(self.nc, dtype=np.float)
        for i, l in enumerate(labels):
            c = centroids[l, :]
            d = self.dist.pairwise_single(X[i, :], c)
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
                        d = self.dist.pairwise_single(centroids[i, :], centroids[j, :])
                        if not (d < self.tol) and d < inter_cluster[i]:
                            inter_cluster[i] = d
            self.cluster_deviation = inter_cluster / max(inter_cluster)

    def decision_function(self, X):
        """Compute the anomaly scores.

        Parameters
        ----------
        X : np.ndarray
            The input instances.

        Returns
        -------
        scores : np.ndarray
            The anomaly scores of the input instances.
        """

        n, _ = X.shape

        # predict the cluster labels + distances to the clusters
        labels = self.base_clusterer.predict(X)
        # each dimension is the distance to the cluster centers (n_samples, n_clusters)
        distances = self.base_clusterer.transform(X)

        # compute scores
        scores = np.zeros(n, dtype=float)
        for i, l in enumerate(labels):
            if self.max_intra_cluster[l] < self.tol:
                point_deviation = 1.0
            else:
                point_deviation = distances[i, l] / self.max_intra_cluster[l]
            scores[i] = (
                point_deviation * self.cluster_deviation[l]
            ) / self.cluster_sizes[l]

        return scores
