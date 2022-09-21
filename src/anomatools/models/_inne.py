"""
Isolation nearest neighbor ensembles.

Reference:
    T. R. Bandaragoda, K. Ming Ting, D. Albrecht, F. T. Liu, Y. Zhu, and J. R. Wells.
    Isolation-based anomaly detection using nearest-neighbor ensembles.
    In Computational Intelligence, vol. 34, 2018, pp. 968-998.

:author: Vincent Vercruyssen (2022)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import math
import numpy as np

from copy import deepcopy
from sklearn.base import BaseEstimator

from ._base import BaseDetector
from .scalers import SquashScaler


# ----------------------------------------------------------------------------
# iNNE detector
# ----------------------------------------------------------------------------


class iNNE(BaseEstimator, BaseDetector):
    """
    The Isolation-based Nearest Neighbor Ensemble anomaly detector
    constructs hyperspheres around a limited set of instances in
    the dataset. To detect anomalies, distances to the hypersphere
    centers are computed.

    Parameters
    ----------
    n_members : int, optional
        Number of estimators to construct.

    sample_size : int, optional
        Number of instances in each subsample.
        If sample size is None, set to square root of N.

    scaler : object, optional
        Scaling applied to the anomaly scores.

    metric : str or callable, optional
        The distance metric. Currently, only sklearn metrics.

    metric_params : dict, optional
        Additional keyword arguments for the metric function.

    verbose : bool, optional
        Verbosity.

    Attributes
    ----------
    ensemble_ : list of hyperspheres
        The fitted hypersphere objects.
    """

    def __init__(
        self,
        n_members=100,
        sample_size=None,
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

        self.n_members = n_members
        self.sample_size = sample_size

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

        # compute sample size
        n, _ = X.shape
        if self.sample_size is None:
            self.sample_size = max(1, int(math.floor(np.sqrt(n))))
        self.sample_size = min(self.sample_size, n)

        # fit the ensemble members
        self.ensemble_ = []
        for _ in range(self.n_members):
            # random sample (same point can be sampled multiple times)
            if sample_weight is None:
                ixs = np.random.choice(n, self.sample_size)
            else:
                assert len(sample_weight) == n, "sample_weight should be same size as X"
                sample_weight /= np.sum(sample_weight)
                ixs = np.random.choice(n, self.sample_size, p=sample_weight)

            # drop duplicates
            ixs = np.unique(ixs)
            sphere = HyperSphere(X[ixs, :], self.dist, self.tol)
            self.ensemble_.append(sphere)

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
        scores = np.zeros(n, dtype=float)

        # iterate over the ensemble
        for sphere in self.ensemble_:
            sphere_scores = np.ones(n, dtype=float)

            s_dists, s_ixs = sphere.nn.search_neighbors(X, k=sphere.n - 1)

            for i in range(n):
                cr = sphere.radii[s_ixs[i, :].flatten()]

                # belongs to these spheres
                ix_m = np.where(s_dists[i, :].flatten() <= cr)[0]

                # does not belong to sphere
                if len(ix_m) == 0:
                    continue

                # sphere with smallest radius
                ixs = np.argmin(cr[ix_m])
                ns = s_ixs[i, ixs]
                sphere_scores[i] = sphere.scores[ns]

            # update overal anomaly scores
            scores += sphere_scores

        scores /= len(self.ensemble_)
        return scores


# ----------------------------------------------------------------------------
# single member of the iNNE ensemble
# ----------------------------------------------------------------------------


class HyperSphere:
    """
    Construct a single hypershpere.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The training instances.

    dist : DistanceFun
        Allows distance computations.

    tol : float
        Tolerance.
    """

    def __init__(self, X, dist, tol):
        # constructs hypersphere
        self.n = X.shape[0]

        # compute distance matrix (over the sample!)
        self.nn = deepcopy(dist)
        self.nn.fit(X)
        nn_dists, nn_ixs = self.nn.search_neighbors(X, k=2)

        # radii (to nearest neighbor), 0 = point itself
        self.radii = nn_dists[:, 1].flatten() + tol

        # isolation scores
        self.scores = 1.0 - (self.radii[nn_ixs[:, 1].flatten()] / self.radii)
