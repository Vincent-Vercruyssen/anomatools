"""
Base anomaly detector class. The class provideds a blueprint for the anomaly detection methods.

:author: Vincent Vercruyssen (2022)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from abc import abstractmethod, ABCMeta

from ..utils.config import Config
from ..utils.utils import DistanceFun
from ..utils.validation import check_arrays_X_y, check_is_fitted
from .scalers import SquashScaler


# -------------------------------------
# base
# -------------------------------------


class BaseDetector(metaclass=ABCMeta):
    """
    Abstract class for the anomaly detection algorithms.

    Parameters
    ----------
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
    fitted_ : bool
        Is the detector fitted or not.
    """

    @abstractmethod
    def __init__(
        self, scaler=SquashScaler(), metric="euclidean", metric_params={}, verbose=False
    ):

        self.scaler = scaler
        self.metric = metric
        self.metric_params = metric_params
        self.dist = DistanceFun(metric, metric_params)
        self.verbose = verbose
        self.tol = Config.tolerance

    # ----------------------------- ABSTRACT -----------------------------
    @abstractmethod
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

        pass

    @abstractmethod
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
        """

        pass

    # ----------------------------- METHODS -----------------------------
    def fit_predict(self, X, y=None, sample_weight=None):
        """Fit and predict the anomaly detector.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            The training instances.

        y : np.ndarray or pd.DataFrame, optional
            The training instance class labels as ints.

        sample_weight : np.ndarray, optional
            The sample weights of the training instances.

        Returns
        -------
        scores : np.ndarray
            The anomaly scores of the input instances.
        """

        self.fit(X, y, sample_weight)
        return self.predict(X)

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model on the data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            The training instances.

        y : np.ndarray or pd.DataFrame, optional
            The training instance class labels as ints.

        sample_weight : np.ndarray, optional
            The sample weights of the training instances.

        Returns
        -------
        self : object
            Fitted detector.
        """

        X, y = check_arrays_X_y(X, y)

        # fit detector + scaler
        self._train(X, y, sample_weight)
        train_scores = self.decision_function(X)
        _ = self.scaler.fit(S=train_scores)
        self.fitted_ = True
        return self

    def predict(self, X):
        """Predict the anomaly labels.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            The input instances.

        Returns
        -------
        labels : np.ndarray
            Anomaly labels, -1 = normal, 1 = anomaly.
        """

        check_is_fitted(self, self.fitted_)
        X, _ = check_arrays_X_y(X, None)

        # labels
        scores = self.decision_function(X)
        probs = self.scaler.transform(scores)
        labels = np.ones(len(X), dtype=int) * -1
        labels[probs > self.scaler.get_threshold()] = 1
        return labels

    def predict_proba(self, X):
        """Predict the anomaly probabilities of the data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            The input instances.

        Returns
        -------
        probs : np.ndarray
            Return [normal prob, anomaly prob], following PyOD convention.
        """

        check_is_fitted(self, self.fitted_)
        X, _ = check_arrays_X_y(X, None)

        # labels
        scores = self.decision_function(X)
        anom_prob = self.scaler.transform(scores)
        norm_prob = 1.0 - anom_prob
        probs = np.array((norm_prob, anom_prob)).T

        return probs
