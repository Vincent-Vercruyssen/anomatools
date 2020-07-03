# -*- coding: UTF-8 -*-
"""

Functions for fast distance computations.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import NearestNeighbors, BallTree
try:
    from dtaidistance import dtw
except ImportError:
    raise ImportError('Import ERROR: `dtaidistance` python library not installed. Without it, you cannot use `metric=dtw`')


# ----------------------------------------------------------------------------
# helper classes
# ----------------------------------------------------------------------------

class DistanceFun:
    """ Abstract class for the anomaly detection algorithms.

    Parameters
    ----------
    metric : string (default=euclidean)
        Distance metric for constructing the BallTree.
        Can be any of sklearn.neighbors.DistanceMetric methods or 'dtw'.

    TODO: optimize for speed (e.g., multiple processors)
    TODO: datatype of data to np.double
    TODO: speed-up using cdist and pdist (scipy.spatial.distance)
    """
    def __init__(self,
                 metric='euclidean',
                 verbose=False):

        # instantiate distance metric
        self.metric_name = str(metric).lower()
        if self.metric_name == 'dtw':
            if dtw.try_import_c():
                raise Exception(metric, 'C-library of `dtaidistance` not available, please build the C code')
            self.metric = DistanceMetric.get_metric(dtw.distance_fast)
        else:
            try:
                self.metric = DistanceMetric.get_metric(self.metric_name)
            except:
                raise ValueError(metric, 'is not an accepted distance metric')

    def pairwise_single(self, x1, x2):
        """ Return the pairwise distance between two examples x1 and x2. """
        return self.metric.pairwise(x1.astype(np.double).reshape(1, -1),
            x2.astype(np.double).reshape(1, -1))[0][0]

    def pairwise_multiple(self, X, Y=None):
        """ Return the pairwise distance matrix between X (and optionally Y). """
        if Y is None:
            return self.metric.pairwise(X.astype(np.double))
        return self.metric.pairwise(X.astype(np.double), Y.astype(np.double))

    def fit(self, X):
        """ Fit nearest neighbors to the data. """
        if self.metric_name == 'dtw':
            # dtw is not a metric! Brute force is unfortunately the only option
            self.nbrs_ = NearestNeighbors(algorithm='brute', metric=dtw.distance_fast)
            self.nbrs_.fit(X.astype(np.double))
        else:
            # use only methods that are true metrics
            self.tree_ = BallTree(X.astype(np.double), leaf_size=40, metric=self.metric)
        return self

    def search_neighbors(self, X, k=10, exclude_first=False):
        """ Search the nearest neighbors. """
        if self.metric_name == 'dtw':
            D, I = self.nbrs_.kneighbors(X.astype(np.double), n_neighbors=k+1, return_distance=True)
        else:
            D, I = self.tree_.query(X.astype(np.double), k=k+1, return_distance=True, dualtree=True)
        if exclude_first:
            return D[:, 1:], I[:, 1:]
        return D[:, :k], I[:, :k]

    def search_radius(self, X, r=None):
        """ Search the radius for neighbors. """
        if self.metric_name == 'dtw':
            Dr, Ir = self.nbrs_.radius_neighbors(X, radius=r, return_distance=True)
        else:
            Ir, Dr = self.tree_.query_radius(X, r=r, return_distance=True, count_only=False)
        return Ir, Dr
        