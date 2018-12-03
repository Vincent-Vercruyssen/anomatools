""" Isolation Forests.

Reference:
    Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008, December). Isolation forest.
    In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE.

"""

# Authors: Vincent Vercruyssen, 2018.

import numpy as np

from sklearn.ensemble import IsolationForest

from .BaseDetector import BaseDetector
from ..utils.validation import check_X_y


# -------------
# CLASSES
# -------------

class iForest(BaseDetector):
    """ Isolation Forest anomaly detection.

    Parameters
    ----------
    n_estimators : int (default=100)
        Number of decision tree estimators.

    contamination : float (default=0.1)
        Estimate of the expected percentage of anomalies in the data.

    ... (see sklearn implementation for other parameters)

    Comments
    --------
    - Implementation with support for out-of-sample setting.
    """

    def __init__(self, n_estimators=100, max_samples='auto', max_features=1.0, contamination=0.1,
                 tol=1e-8, verbose=False):
        super(iForest, self).__init__()

        self.n_estimators = int(n_estimators)
        if isinstance(max_samples, str):
            self.max_samples = str(max_samples)
        else:
            self.max_samples = int(max_samples)
        if isinstance(max_features, int):
            self.max_features = int(max_features)
        else:
            self.max_features = float(max_features)
        self.contamination = float(contamination)

        self.tol = float(tol)
        self.verbose = bool(verbose)

    def fit_predict(self, X, y=None):
        """ Fit the model to the training set X and returns the anomaly score
            of the instances in X.

        :param X : np.array(), shape (n_samples, n_features)
            The samples to compute anomaly score w.r.t. the training samples.
        :param y : np.array(), shape (n_samples), default = None
            Labels for examples in X.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        X, y = check_X_y(X, y)

        return self.fit(X, y).predict(X)

    def fit(self, X, y=None):
        """ Fit the model using data in X.

        :param X : np.array(), shape (n_samples, n_features)
            The samples to compute anomaly score w.r.t. the training samples.
        :param y : np.array(), shape (n_samples), default = None
            Labels for examples in X.

        :returns self : object
        """

        X, y = check_X_y(X, y)
        n, _ = X.shape

        self.mdl = IsolationForest(n_estimators=self.n_estimators,
                                   max_samples=self.max_samples,
                                   max_features=self.max_features,
                                   contamination=self.contamination,
                                   verbose=0)
        self.mdl.fit(X)

        return self

    def predict(self, X):
        """ Compute the anomaly score + predict the label of instances in X.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        X, y = check_X_y(X, None)
        n, _ = X.shape

        # average anomaly score for each sample
        if_scores = self.mdl.decision_function(X) * -1  # higher = more anomalous
        y_score = (if_scores - min(if_scores)) / (max(if_scores) - min(if_scores))

        # prediction threshold + absolute predictions
        self.threshold = np.sort(y_score)[int(n * (1.0 - self.contamination))]
        y_pred = np.ones(n, dtype=float)
        y_pred[y_score < self.threshold] = -1

        return y_score, y_pred
