"""
Semi-Supervised Detection of Anomalies.

Reference:
    V. Vercruyssen, W. Meert, G. Verbruggen, K. Maes, R. Baumer, J. Davis.
    Semi-supervised anomaly detection with an application to water analytics.
    In IEEE International Conference on Data Mining, Singapore, 2018, pp. 527–536.

:author: Vincent Vercruyssen (2022)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np
import scipy.stats as sps

from sklearn.base import BaseEstimator

from ._base import BaseDetector
from ._knno import kNNO
from .scalers import NoneScaler


# ----------------------------------------------------------------------------
# SSDO detector
# ----------------------------------------------------------------------------


class SSDO(BaseEstimator, BaseDetector):
    """
    The SSDO algorithm first computes an unsupervised score. Then, it
    updates the unsupervised score based on the given labels.

    Parameters
    ----------
    k : int, optional
        Controls how many instances are updated by propagating
        the label of a single labeled instance.

    alpha : float, optional
        User influence parameter that controls the weight given to the
        unsupervised and label propragation components of an instance's
        anomaly score. Higher = more weight to supervised component.

    base_detector : object, optional
        Base unsupervised detector. Only works with objects that have the
        'predict_proba()' implemented (e.g., PyOD or anomatools).

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
        k=30,
        alpha=2.3,
        base_detector=kNNO(),
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
        self.alpha = alpha
        self.base_detector = base_detector

    # ----------------------------- OVERRIDE -----------------------------
    def _train(self, X, y=None, sample_weight=None):
        """Fit the detector.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            The training instances.

        y : np.ndarray or pd.DataFrame, optional
            The training instance class labels as ints.

        sample_weight : np.ndarray, optional
            The sample weights of the training instances.
        """

        # store the labeled instances
        if y is None:
            y = np.zeros(len(X))
        ixl = np.where(y != 0)[0]
        self.feedback_ = y[ixl].copy()
        self.X_feedback_ = X[ixl, :].copy()

        # fit the base detector
        # applies its own scaling internally
        self.base_detector.fit(X, sample_weight=sample_weight)

        # compute eta parameter
        self.eta_ = self._compute_eta(X)

    def decision_function(self, X):
        """Compute the anomaly scores.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            The input instances.

        Returns
        -------
        scores : np.ndarray
            The anomaly scores of the input instances.
            No further scaling is required here.
        """

        # compute the anomaly probabilities in [0, 1]
        # applies its own scaling internally
        scaled_prior = self.base_detector.predict_proba(X)[:, 1]

        # no feedback
        if len(self.feedback_) == 0:
            return scaled_prior

        # feedback: compute posterior
        posterior = self._compute_posterior(X, scaled_prior)

        return posterior

    # ----------------------------- INTERNAL -----------------------------
    def _compute_posterior(self, X, scaled_prior):
        """Update the prior score with label propagation.

        Parameters
        ----------
        X : np.ndarray
            The input instances.

        scaled_prior : np.ndarray
            Scaled unsupervised prior of the input instances.

        Returns
        -------
        posterior : np.ndarray of shape (n_samples,)
            Posterior anomaly score between 0 and 1.
        """

        # labeled examples (normals, anomalies)
        ixn = np.where(self.feedback_ == -1.0)[0]
        ixa = np.where(self.feedback_ == 1.0)[0]

        # compute limited distance matrices (to normals, to anomalies)
        if len(ixn) > 0:
            Dnorm = self.dist.pairwise_multiple(X, self.X_feedback_[ixn, :])
        if len(ixa) > 0:
            Danom = self.dist.pairwise_multiple(X, self.X_feedback_[ixa, :])

        # compute posterior
        posterior = np.zeros(len(scaled_prior), dtype=float)
        for i in range(len(scaled_prior)):
            # weighted distance to anomalies & normals
            da = 0.0
            dn = 0.0
            if len(ixa) > 0:
                da = np.sum(self._ssdo_squashing_function(Danom[i, :], self.eta_))
            if len(ixn) > 0:
                dn = np.sum(self._ssdo_squashing_function(Dnorm[i, :], self.eta_))

            # posterior
            z = 1.0 / (1.0 + self.alpha * (da + dn))
            posterior[i] = z * (scaled_prior[i] + self.alpha * da)

        return posterior

    def _compute_eta(self, X):
        """Compute the eta parameter.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input instances.

        Returns
        -------
        eta : float
            Eta parameter is the harmonic mean of the k-distances.
        """

        # fit neighbors
        self.dist.fit(X)

        # find nearest neighbors
        D, _ = self.dist.search_neighbors(X, k=self.k, exclude_first=True)
        d = D[:, -1].flatten()

        # compute eta as the harmonic mean of the k-distances
        filler = np.min(d[d > 0.0])
        d[d == 0.0] = filler
        eta = sps.hmean(d)

        if eta < self.tol:
            eta = self.tol

        return eta

    def _ssdo_squashing_function(self, x, gamma):
        """Compute the value of x under squashing function."""
        return np.exp(np.log(0.5) * np.power(x / gamma, 2))
