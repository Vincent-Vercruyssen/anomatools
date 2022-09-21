"""
Functions for fast distance computation.

:author: Vincent Vercruyssen (2022)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from sklearn.metrics import DistanceMetric
from sklearn.neighbors import NearestNeighbors

try:
    from dtaidistance import dtw

    DTW_AVAILABLE = True
except ImportError:
    print(
        f"WARNING: `dtaidistance` python library not installed. Without it, you cannot use `metric=dtw`"
    )
    DTW_AVAILABLE = False


# ---------------------------------------------------------------------------------
# classes
# ---------------------------------------------------------------------------------


class DistanceFun:
    """Custom distance computations.

    Parameters
    ----------
    metric : string, optional
        Distance metric name for constructing the BallTree.
        Can be any of sklearn.neighbors.DistanceMetric methods or 'dtw'.

    metric_params : dict, optional
        Additional keyword arguments for the metric function.
    """

    def __init__(self, metric="euclidean", metric_params={}):

        self.metric = metric
        self.metric_params = metric_params

        if self.metric == "dtw":
            if DTW_AVAILABLE:
                try:
                    self.dist = DistanceMetric.get_metric(dtw.distance_fast)
                except Exception as e:
                    raise Exception(f"ERROR: {e}")
            else:
                raise ValueError(
                    "ERROR: `dtw` is not available, install `dtaidistance` library"
                )
        else:
            try:
                self.dist = DistanceMetric.get_metric(self.metric, **self.metric_params)
            except:
                raise ValueError(
                    f"ERROR: `{self.name}` is not an accepted distance metric"
                )

    def pairwise_single(self, x, y):
        return self.dist.pairwise(
            x.astype(np.double).reshape(1, -1), y.astype(np.double).reshape(1, -1)
        )[0][0]

    def pairwise_multiple(self, X, Y=None):
        if Y is None:
            return self.dist.pairwise(X.astype(np.double))
        return self.dist.pairwise(X.astype(np.double), Y.astype(np.double))

    def fit(self, X):
        if self.metric == "dtw":
            # dtw is not a metric! Brute force is unfortunately the only option
            self.nbrs = NearestNeighbors(algorithm="brute", metric=dtw.distance_fast)
        else:
            self.nbrs = NearestNeighbors(metric=self.metric, **self.metric_params)
        self.nbrs.fit(X.astype(np.double))
        return self

    def search_neighbors(self, X, k, exclude_first=False):
        # nn search for X contains points themselves if X was also used to fit
        D, I = self.nbrs.kneighbors(X.astype(np.double), k + 1, return_distance=True)
        if exclude_first:
            return D[:, 1:], I[:, 1:]
        return D[:, :k], I[:, :k]

    def search_radius(self, X, r):
        # r can be an array (one radius per example)
        D, I = self.nbrs.radius_neighbors(X, radius=r, return_distance=True)
        return D, I
