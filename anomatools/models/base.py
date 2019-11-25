# -*- coding: UTF-8 -*-
"""

Base class for all models.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from abc import abstractmethod, ABCMeta
from scipy.special import erf
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import DistanceMetric


# ----------------------------------------------------------------------------
# BaseDetector
# ----------------------------------------------------------------------------


class BaseDetector(metaclass=ABCMeta):
    """ Abstract class for the anomaly detection algorithms.

    Parameters
    ----------
    contamination : float in (0, 0.5), optional (default=0.1)
        The expected proportion of anomalies in the dataset.

    metric : string (default=euclidean)
        Distance metric used in the detector.

    tol : float in (0, +inf), optional (default=1e-10)
        The tolerance.

    verbose : bool, optional (default=False)
        Verbose or not.

    Attributes
    ----------
    scores_ : np.array of shape (n_samples,)
        The anomaly scores of the training data (higher = more abnormal).
    
    threshold_ : float
        The cutoff threshold on the anomaly score separating the normals
        from the anomalies. This is based on the `contamination` parameter.

    labels_ : np.array of shape (n_samples,)
        Binary anomaly labels (-1 = normal, +1 = anomaly).
    """

    @abstractmethod
    def __init__(self,
                 contamination=0.1,
                 metric='euclidean',
                 tol=1e-10,
                 verbose=False):
        super().__init__()

        # contamination
        if not(0.0 < contamination <= 0.5):
            raise ValueError(contamination, 'is not a float in (0.0, 0.5]')
        self.c = float(contamination)

        # distance metric
        try:
            self.metric = DistanceMetric.get_metric(metric)
        except:
            raise ValueError(metric, 'is not an accepted distance metric')

        self.tol = float(tol)
        self.verbose = bool(verbose)
        
        # internal
        self.derived_squashed_ = False

    # must be implemented by derived classes
    @abstractmethod
    def fit(self, X, y=None):
        """ Fit the model on data X.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances. 
        y : np.array of shape (n_samples,), optional (default=None)
            The ground truth of the input instances.

        Returns
        -------
        self : object
        """

        pass

    @abstractmethod
    def decision_function(self, X):
        """ Compute the anomaly scores of X.
        
        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.

        Returns
        -------
        scores : np.array of shape (n_samples,)
            The anomaly scores of the input instances.
        """
        
        pass

    # must NOT be implemented by derived classes
    def fit_predict(self, X, y=None, *args, **kwargs):
        """ Fit the model on data X.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances. 
        y : np.array of shape (n_samples,), optional (default=None)
            The ground truth of the input instances.

        Returns
        -------
        labels : np.array of shape (n_samples,)
            The labels (-1, +1) of the input instances.
        """
        
        self.fit(X, y, *args, **kwargs)
        return self.labels_

    def predict(self, X, *args, **kwargs):
        """ Predict the labels of X.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.

        Returns
        -------
        labels : np.array of shape (n_samples,)
            The labels (-1, +1) of the input instances.
        """

        check_is_fitted(self, ['scores_', 'threshold_', 'labels_'])

        # call the decision function
        scores = self.decision_function(X, *args, **kwargs)

        # compute the labels
        labels = np.ones(len(scores), dtype=int)
        labels[scores <= self.threshold_] = -1
        return labels

    def predict_proba(self, X, method='squash', *args, **kwargs):
        """ Predict the anomaly probabilities of X using the fitted model.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        method : str, optional (default='squashing')
            The method to compute the probabilities:
            'linear'    --> min-max scaling of the scores
            'squash'    --> using an exponential squashing function
            'unify'     --> use unifying scores
            'none'      --> return scores as is

        Returns
        -------
        probs : np.array of shape (n_samples,)
            The anomaly probabilities of the input instances.
        """

        check_is_fitted(self, ['scores_', 'threshold_', 'labels_'])

        # call the decision function
        test_scores = self.decision_function(X, *args, **kwargs)

        # compute the class probabilities
        probs = np.zeros([X.shape[0], 2])
        if method.lower() == 'linear':
            probs[:, 1] = (test_scores - self.min_) / (self.max_ - self.min_)
        elif method.lower() == 'squash':
            probs[:, 1] = self._squashing_function(test_scores, self.threshold_)
        elif method.lower() == 'unify':
            # from: https://github.com/yzhao062/pyod/blob/master/pyod/models/base.py
            pre_erf_score = (test_scores - self.m_) / (self.s_ * np.sqrt(2))
            erf_score = erf(pre_erf_score)
            probs[:, 1] = erf_score.clip(0, 1).ravel()
        elif method.lower() == 'none':
            return test_scores
        else:
            raise ValueError(method, 'is not in [linear, squash, unify, none]')
        
        probs[:, 0] = 1.0 - probs[:, 1]
        return probs

    def _process_anomaly_scores(self):
        """ Compute the relevant attributes of the training anomaly scores.
        
        Comments
        --------
        1. Needs the training scores!
        """

        check_is_fitted(self, ['scores_'])

        # compute statistics on training scores
        self.threshold_ = np.percentile(self.scores_, 100*(1.0-self.c)) + self.tol
        self.m_ = np.mean(self.scores_)
        self.s_ = np.std(self.scores_)
        self.min_ = min(self.scores_)
        self.max_ = max(self.scores_)

        self._scores_to_labels()

    def _scores_to_labels(self):
        """ Transform the scores to discrete labels.
        """

        check_is_fitted(self, ['scores_'])

        self.labels_ = np.ones(len(self.scores_), dtype=int)
        self.labels_[self.scores_ <= self.threshold_] = -1

    def _squashing_function(self, x, gamma):
        """ Compute the squashed x values.
        """

        return 1.0 - np.exp(np.log(0.5) * np.power(x / gamma, 2))    
