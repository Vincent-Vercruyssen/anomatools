# -*- coding: UTF-8 -*-
"""

Constrained KMeans.

Reference:
    https://github.com/Behrouz-Babaki/COP-Kmeans

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import math
import random
import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator

from .utils import DistanceFun


# ----------------------------------------------------------------------------
# COPKMeans
# ----------------------------------------------------------------------------

class COPKMeans(BaseEstimator):
    """
    Parameters
    ----------
    n_clusters : int (default=10)
        Number of clusters used for the COP k-means clustering algorithm.

    init : str (default='kmpp')
        Initialization method for the cluster centra.

    n_init : int (default=3)
        Number of initializations.
    
    max_iter : int (default=300)
        Maximum iterations for the algorithm.

    metric : string (default=euclidean)
        Distance metric for constructing the BallTree.
        Can be any of sklearn.neighbors.DistanceMetric methods or 'dtw'

    chunk_size : int (default=2000)
        Size of each chunck to recompute the cluster centra.
    """

    def __init__(self,
                 n_clusters=10,
                 init='kmpp',
                 n_init=3,
                 max_iter=300,
                 metric='euclidean',
                 chunk_size=2000,
                 tol=1e-10,
                 verbose=False):
        super().__init__()

        # initialize parameters
        self.k = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.chunk_size = chunk_size
        self.sample_tol = 1e-10
        self.n_samples = None
        self.D_matrix_ = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.dist = DistanceFun(metric)

    def fit_predict(self, X, must_link=np.array([]), cannot_link=np.array([])):
        """ Fit COP Kmeans clustering to the data given the constraints.

        :param X: np.array()
            2D data array to be clustered.
        :param must_link: np.array() [n_mlcs, 2]
            Indices of the must link datapoints.
        :param cannot_link : np.array() [n_clcs, 2]
            Indices of cannot link datapoints.

        :returns cluster_centers_: np.array()
            Cluster centroids.
        :returns labels_: np.array()
            Labels of the centroids to which the points belong.
        """

        return self.fit(X, must_link, cannot_link)._predict()

    def fit(self, X, must_link=np.array([]), cannot_link=np.array([])):
        """ Fit COP Kmeans clustering to the data given the constraints.

        :param X: np.array()
            2D data array to be clustered.
        :param must_link: np.array() [n_mlcs, 2]
            Indices of the must link datapoints.
        :param cannot_link : np.array() [n_clcs, 2]
            Indices of cannot link datapoints.
        """

        self.e = self._calc_tolerance(X)  # also computes the n_samples
        self.k = min(self.k, self.n_samples)

        # compute the transitive closure
        ml_graph, cl_graph = self._transitive_closure(must_link, cannot_link, self.n_samples)

        # fit clustering multiple times
        self.inertia_ = np.inf
        for i in range(self.n_init):
            k = self.k - 1
            success = False
            while not success:
                k += 1
                centers, labels, fk = self._single_cop_kmeans(X, k, ml_graph, cl_graph)
                if len(labels) > 0:
                    success = True

            inertia = self._calc_inertia(X, centers, labels)
            if inertia < self.inertia_:
                self.cluster_centers_ = centers
                self.labels_ = labels
                self.inertia_ = inertia
                self.n_clusters = fk

        # print('fitting copkmeans done')
        return self

    def predict(self, X, include_distances=False):
        """ Predict the cluster labels of the given points.

        :param X: np.array()
            2D data array to be clustered.
        :param include_distances: bool
            Return the distances to the centers.

        :returns cluster_centers_: np.array()
            Cluster centroids.
        :returns labels_: np.array()
            Labels of the centroids to which the points belong.
        :returns distances: np.array()
            Distances to the cluster centers.
        """

        n, _ = X.shape
        labels = np.zeros(n, dtype=int)
        if include_distances:
            distances = np.zeros(n, dtype=np.float)

        # compute labels in chunks
        n_chunks = math.ceil(n / self.chunk_size)
        for i in range(n_chunks):
            X_chunk = X[i*self.chunk_size:(i+1)*self.chunk_size, :]
            # compute the distance to each cluster centroid
            D_matrix = self.dist.pairwise_multiple(X_chunk, self.cluster_centers_)
            # select the centroid with the lowest distance
            cc_indices = np.argsort(D_matrix, axis=1)
            labels[i*self.chunk_size:(i+1)*self.chunk_size] = cc_indices[:, 0].T
            # return the distances
            if include_distances:
                dists = np.sort(D_matrix, axis=1)
                distances[i*self.chunk_size:(i+1)*self.chunk_size] = dists[:, 0].T

        if include_distances:
            return self.cluster_centers_, labels, distances
        return self.cluster_centers_, labels

    def _predict(self):
        """ Predict the cluster labels based on the centroids.

        :returns centers: np.array()
            Cluster centroids.
        :returns cluster_labels: np.array()
            Labels of the centroids to which the points belong.
        """

        return self.cluster_centers_, self.labels_

    def _single_cop_kmeans(self, data, k, ml_graph, cl_graph):
        """ Core COP Kmeans algorithm.

        :param data: np.array()
            2D data array to be clustered.
        :param k: int
            Number of clusters.
        :param ml_graph: dict()
            Must links for each datapoint.
        :param cl_graph: dict()
            Cannot links for each datapoint.

        :returns centers: np.array()
            2D array containing the cluster centers.
        :returns labels: np.array()
            Cluster label for each datapoint.
        """

        # initialize cluster centers
        centers = self._intialize_centers(data, k)

        # iterate untill convergence
        for _ in range(self.max_iter):
            old_centers = centers.copy()
            labels = np.ones(self.n_samples, dtype=int) * -1

            # 1. compute the distance matrix
            self.D_matrix_ = self.dist.pairwise_multiple(data, old_centers)

            # 2. assign the points to clusters
            for i in range(self.n_samples):
                if labels[i] == -1:
                    cc_indices = np.argsort(self.D_matrix_[i, :])
                    found_cluster = False
                    for cc in cc_indices:
                        if not self._violate_constraints(i, cc, labels, ml_graph, cl_graph):
                            found_cluster = True
                            labels[i] = int(cc)
                            for j in ml_graph[i]:
                                labels[j] == int(cc)
                            break

                    if not found_cluster:
                        return np.array([]), np.array([]), k

            # 3. recompute the centers
            centers, k = self._compute_centers(data, labels, k)

            # 4. stopping criterion
            early_stopping = self._calc_stopping_criterion(centers, old_centers)
            if early_stopping:
                break

        # return centers and labels
        return centers, labels, k

    def _intialize_centers(self, data, k):
        """ Initialize the cluster centers.

        :param data: np.array()
            2D array containing the data.
        :param k: int
            Number of clusters.

        :returns centers: np.array()
            2D array with cluster centers.
        """

        if self.init == 'kmpp':
            # k-means ++ initialization
            ix_ = np.random.choice(self.n_samples, 1)
            centers = data[ix_, :]
            probs = np.ones(self.n_samples)

            while len(centers) < k:
                D2 = np.array([min([self.dist.pairwise_single(x, c)**2 for c in centers]) for x in data])
                if D2.sum() == 0:
                    break
                else:
                    probs = D2 / D2.sum()
                    cumprobs = probs.cumsum()
                    r = random.random()
                    ix = np.where(cumprobs >= r)[0][0]
                    centers = np.vstack((centers, data[ix, :]))

        else:
            # random initialization
            idx_c = np.random.choice(self.n_samples, k, replace=False)
            centers = data[idx_c, :]

        return centers

    def _compute_centers(self, data, labels, k):
        """ Compute the new cluster centers.

        :param data: np.array()
            2D array with the datapoints.
        :param labels: np.array()
            New cluster labels assigned to data points.
        :param k: int
            Number of clusters.

        :returns new_centers: np.array()
            Updated cluster centers.
        """

        new_centers = np.zeros((k, self.n_dim))
        count_vector = np.zeros(k)
        for i, l in enumerate(labels):
            new_centers[l, :] += data[i, :]
            count_vector[l] += 1

        ix = np.where(count_vector > 0)[0]
        new_centers = new_centers[ix, :]
        count_vector = count_vector[ix]
        new_centers = new_centers / count_vector[:, None]

        return new_centers, len(new_centers)

    def _violate_constraints(self, data_index, c_index, labels, ml_graph, cl_graph):
        """ Check if the cluster assignment does not violate constraints.

        :param data_index: int
            Index of the point.
        :param c_index: int
            Index of the cluster to which point is assigned.
        :param labels: np.array()
            Label assigment so far.
        :param ml_graph: dict()
            Must links for each datapoint.
        :param cl_graph: dict()
            Cannot links for each datapoint.

        :returns violate: boolean
            Violate constraint or not.
        """

        for i in ml_graph[data_index]:
            if labels[i] != -1 and labels[i] != c_index:
                return True

        for i in cl_graph[data_index]:
            if labels[i] == c_index:
                return True

        return False

    def _calc_stopping_criterion(self, centers, old_centers):
        """ Determine the stopping criterion.

        :param centers: np.array()
            New cluster centers.
        :param old_centers: np.array()
            Old cluster centers.

        :returns stopping_criterion: boolean
            Stop or continue.
        """

        d = 0.0
        for i in range(len(centers)):
            d += self.dist.pairwise_single(centers[i, :], old_centers[i, :])
        if d < self.e:
            return True
        else:
            return False

    def _calc_tolerance(self, data):
        """ Compute the tolerance.

        :param data: np.array()
            2D array containing the data.

        :returns e: float
            Tolerance.
        """

        self.n_samples, self.n_dim = data.shape
        variances = np.var(data, axis=0)

        return self.sample_tol * sum(variances) / self.n_dim

    def _transitive_closure(self, ml, cl, n):
        """ Compute transitive closure. """

        ml_graph = dict()
        cl_graph = dict()
        for i in range(n):
            ml_graph[i] = set()
            cl_graph[i] = set()

        def add_both(d, i, j):
            d[i].add(j)
            d[j].add(i)

        for (i, j) in ml:
            add_both(ml_graph, i, j)

        def dfs(i, graph, visited, component):
            visited[i] = True
            for j in graph[i]:
                if not visited[j]:
                    dfs(j, graph, visited, component)
            component.append(i)

        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, ml_graph, visited, component)
                for x1 in component:
                    for x2 in component:
                        if x1 != x2:
                            ml_graph[x1].add(x2)
        for (i, j) in cl:
            add_both(cl_graph, i, j)
            for y in ml_graph[j]:
                add_both(cl_graph, i, y)
            for x in ml_graph[i]:
                add_both(cl_graph, x, j)
                for y in ml_graph[j]:
                    add_both(cl_graph, x, y)

        for i in ml_graph:
            for j in ml_graph[i]:
                if j != i and j in cl_graph[i]:
                    raise Exception('inconsistent constraints between %d and %d' % (i, j))

        return ml_graph, cl_graph

    def _calc_inertia(self, data, centers, labels):
        """ Calculate the inertia of the current clustering.

        :param data : np.array() [n_samples, n_features]
            2D array of the data.
        :param centers: np.array() [n_clusters, n_features]
            Cluster centers.
        :param labels: np.array() [n_samples]
            Cluster center to which each sample belongs.

        :returns inertia: float
            Sum of distances of samples to their closest cluster center.
        """

        inertia = 0.0
        for i, l in enumerate(labels):
            inertia += self.dist.pairwise_single(data[i, :], centers[l, :])

        return inertia
