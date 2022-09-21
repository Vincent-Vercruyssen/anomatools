"""
k-Nearest neighbor anomaly detection.

Reference:
    S. Ramaswamy, R. Rastogi, and K. Shim. Efficient algorithms for mining outliers from large data sets.
    In Proceedings of the 2000 ACM SIGMOD international conference on Management of data, vol. 29, no. 2. ACM, 2000, pp. 427–438.

:author: Vincent Vercruyssen (2022)
:license: Apache License, Version 2.0, see LICENSE for details.
"""


import math
import numpy as np

from sklearn.base import BaseEstimator

from ._base import BaseDetector
from .scalers import SquashScaler


# ----------------------------------------------------------------------------
# kNNo detector
# ----------------------------------------------------------------------------


class kNNO(BaseEstimator, BaseDetector):
    """
    The k-Nearest Neighbor anomaly detector detects anomalies by
    computing the distances between neighboring instances. Instances
    that are far away from all other instances are considered as
    anomalies.

    Parameters
    ----------
    k : int, optional
        The number of neighbors for the neighbor queries.

    method : str, optional
        - largest: use the distance to the k'th neighbor as anomaly score
        - mean: use the average distance to all k neighbors
        - median: use the median of the distances to the k neighbors

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
        k=10,
        method="mean",
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

        self.k = k
        self.method = method

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

        self.k = min(self.k, len(X))
        self.dist.fit(X)

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

        D, _ = self.dist.search_neighbors(X, self.k)

        if self.method == "largest":
            scores = D[:, -1].flatten()
        elif self.method == "mean":
            scores = np.mean(D, axis=1)
        elif self.method == "median":
            scores = np.median(D, axis=1)
        else:
            raise NotImplementedError()

        return scores
