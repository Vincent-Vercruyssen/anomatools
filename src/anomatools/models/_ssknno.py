"""
Semi-supervised k-nearest neighbor outlier detection.

Reference:
    V. Vercruyssen, W. Meert, J. Davis.
    Transfer Learning for Anomaly Detection through Localized and Unsupervised Instance Selection.
    In AAAI Conference on Artificial Intelligence, New York, 2020.

:author: Vincent Vercruyssen (2022)
:license: Apache License, Version 2.0, see LICENSE for details.
"""


import numpy as np

from sklearn.base import BaseEstimator

from ._base import BaseDetector
from ._knno import kNNO
from .scalers import NoneScaler, SquashScaler


# ----------------------------------------------------------------------------
# SSkNNO class
# ----------------------------------------------------------------------------


class SSkNNO(BaseEstimator, BaseDetector):
    """Semi-supervised k-nearest neighbors anomaly detection.

    Parameters
    ----------
    k : int, optional
        The number of neighbors for the neighbor queries.

    method : str, optional
        - largest: use the distance to the k'th neighbor as anomaly score
        - mean: use the average distance to all k neighbors
        - median: use the median of the distances to the k neighbors

    supervision : str, optional
        How to compute the supervised score component.
        'loose'     --> use all labeled instances in the set of nearest neighbors
        'strict'    --> use only instances that also count the instance among their neighbors

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
        supervision="loose",
        scaler=NoneScaler(),
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
        self.supervision = supervision
        self.base_detector = kNNO(
            k=k,
            method=method,
            scaler=SquashScaler(),
            metric=metric,
            metric_params=metric_params,
        )

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

        # store the labeled instances
        if y is None:
            y = np.zeros(len(X))
        if len(np.where(y != 0)[0]) > 0:
            self.feedback_ = y.copy()
        else:
            self.feedback_ = np.array([])

        # fit the nn
        self.base_detector.fit(X, sample_weight=sample_weight)

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

        # compute prior distances
        # applies its own scaling internally
        scaled_prior = self.base_detector.predict_proba(X)[:, 1]

        # no feedback
        if len(self.feedback_) == 0:
            return scaled_prior

        # compute posterior
        # collect ALL the nearest neighbors in the radius
        D, _ = self.base_detector.dist.search_neighbors(X, k=self.k)
        nn_radii = D[:, -1].flatten() + self.tol
        D, Ixs = self.base_detector.dist.search_radius(X, nn_radii)

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
                reverse_nn = np.where(ndists <= nn_radii[nixs])[0]
                reverse_nn = np.intersect1d(ixl, reverse_nn)
                Ws = len(reverse_nn) / nn
                # previous: Ws = np.sum(w[ixl]) / np.sum(w)

                # supervised score
                if self.supervision == "loose":
                    ixa = np.where(labels > 0)[0]
                    Ss = np.sum(w[ixa]) / np.sum(w[ixl])
                    # previous: Ss = len(ixa) / len(ixl)
                elif self.supervision == "strict":
                    if len(reverse_nn) > 0:
                        ixa = np.where(labels[reverse_nn] > 0)[0]
                        Ss = np.sum(w[ixa]) / np.sum(w[reverse_nn])
                    else:
                        Ss = 0.0
                else:
                    raise ValueError(self.supervision, "is not in [loose, strict]")

            # supervised plays no role
            else:
                Ss = 0.0
                Ws = 0.0

            # combine supervised and unsupervised
            posterior[i] = (1.0 - Ws) * scaled_prior[i] + Ws * Ss

        return posterior
