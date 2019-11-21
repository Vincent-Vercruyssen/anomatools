# -*- coding: UTF-8 -*-
"""

Semi-Supervised Detection of Anomalies

Reference:
    V. Vercruyssen, W. Meert, J. Davis.
    Transfer Learning for Anomaly Detection through Localized and Unsupervised Instance Selection.
    In AAAI Conference on Artificial Intelligence, New York, 2020.

:author: Vincent Vercruyssen
:year: 2018
:license: Apache License, Version 2.0, see LICENSE for details.

"""

import numpy as np

from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator
from sklearn.neighbors import BallTree


# ----------------------------------------------------------------------------
# SSKNNO class
# ----------------------------------------------------------------------------

class SSKNNO(BaseEstimator):
    """ Semi-supervised k-nearest neighbors anomaly detection.

    Parameters
    ----------
    k : int (default=10)
        Number of nearest neighbors.
    
    nn_concept : str (default=radius)
        How to determine which instances are in the neighborhood of an instance:
        'radius'    --> all instances within a fixed radius
        'set'       --> exact set of first k nearest neighbors

    nn_sup : str (default=loose)
        How to compute the supervised score component.
        'loose'     --> use all labeled instances in the set of nearest neighbors
        'strict'    --> use only instances that also count the instance among their neighbors
    
    contamination : float (default=0.1)
        Estimate of the expected percentage of anomalies in the data.
    """

    def __init__(self,
                 k=10,                      # number of neighbors
                 nn_concept='radius',       # how to compute the nearest neighbors
                 nn_sup='loose',            # how to compute the supervised score
                 contamination=0.1,         # expected proportion of anomalies
                 tol=1e-10,                 # tolerance
                 verbose=False):
        super().__init__()

        # initialize parameters
        self.k = int(k)
        self.nn_concept = str(nn_concept).lower()
        self.nn_sup = str(nn_sup).lower()
        self.c = float(contamination)
        self.tol = float(tol)
        self.verbose = bool(verbose)

    def fit_predict(self, X, y=None):
        """ Fit the model to the training set X and return the label
            of the instances in X (labeled and unlabeled).

        :param X : np.array(), shape (n_samples, n_features)
            The samples to compute label w.r.t. the training samples.
        :param y : np.array(), shape (n_samples), default = None
            Labels for examples in X.

        :returns y_prob : np.array(), shape (n_samples)
            Probability of the label for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Predicted label.
        """

        return self.fit(X, y).predict(X)

    def fit(self, X, y=None):
        """ Fit the model using data in X.

        :param X : np.array(), shape (n_samples, n_features)
            Training samples to fit the model.
        :param y : np.array(), shape (n_samples), default = None
            Labels for examples in X.

        :returns self : object
        """

        # check the input
        if y is None:
            y = np.zeros(len(X))
        X, y = check_X_y(X, y)

        n, _ = X.shape

        # construct the BallTree
        self._tree = BallTree(X, leaf_size=16, metric='euclidean')
        D, _ = self._tree.query(X, k=self.k+1)  # ignoring the instance itself

        # compute the gamma
        outlier_score = D[:, -1].flatten()
        self._gamma = np.percentile(outlier_score, int(100 * (1.0 - self.c))) + self.tol

        # also store the labels, their indices, their radii
        self._labels = y.copy()
        self._radii = D[:, -1].flatten() + self.tol
        
        return self

    def predict(self, X):
        """ Compute the anomaly score + predict the label of instances in X.

        :param X : np.array(), shape (n_samples, n_features)
            Training samples to fit the model.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        # check the input
        X, _ = check_X_y(X, np.zeros(len(X)))

        n, _ = X.shape

        """ TODO: k=self.k unless IN-SAMPLE """
        D, Ixs = self._tree.query(X, k=self.k+1, dualtree=True)
        """ NOTE about the neighborhood conceptualization:

            radius: it can be that > 1 point is identical, in which case selection
            of them in the neighborhood set is random, potentially decreasing the
            impact that labeled instances can have.
        """
        if self.nn_concept == 'radius':
            nn_radii = D[:, -1].flatten()  # = distance to k'th nearest neighbor
            Ixs_radius, D_radius = self._tree.query_radius(
                X, r=nn_radii, return_distance=True, count_only=False)  # additional computation time

        # compute the prior outlier scores
        prior = self._squashing_function(D[:, -1].flatten(), self._gamma)

        # if no labels are available, the posterior == prior
        if len(self._labels) == 0 or len(np.where(self._labels != 0.0)[0]) == 0:
            offset = np.percentile(prior, int(100 * (1.0 - self.c)))
            y_pred = np.ones(n, dtype=int)
            y_pred[prior < offset] = -1
            return prior, y_pred

        # compute the posterior score as a weighted interpolation
        # between the unsupervised and supervised score
        posterior = np.zeros(n, dtype=np.float)
        for i in range(n):
            # k nearest neighbors and indices
            if self.nn_concept == 'set':
                ndists = D[i, :]
                nixs = Ixs[i, :]
                nn = len(ndists)
            elif self.nn_concept == 'radius':
                ndists = D_radius[i]  # could be longer than k
                nixs = Ixs_radius[i]
                nn = len(ndists)  # nn can now change per point
            else:
                raise Exception('INPUT ERROR - unknown `nn_concept`')

            # labels of the neighbors, weights
            labels = self._labels[nixs]
            w = np.power(1.0 / (ndists + self.tol), 2)
            """ Previous version (v0):

                w = 1.0 / (ndists ** 2 + self.tol)
            """

            """ Previous version (v0):

                Check on whether the instance coincides with a labeled instance.
                --> NOT a good check
            """

            # supervised score component
            ixl = np.where(labels != 0.0)[0]
            if len(ixl) > 0:
                # supervised score
                if self.nn_sup == 'loose':
                    ixa = np.where(labels > 0)[0]
                    Ss = np.sum(w[ixa]) / np.sum(w[ixl])
                """ Previous version (v0):
                    
                    The supervised score is the fraction of labeled anomalies
                    in the set of labeled neighbors:

                    Ss = len(ixa) / len(ixl)
                """

                # weight of the supervised component
                # --> the number of labeled instances that also contain this instance as their neighbor
                """ NOTE:

                    'radius' and 'set' have the same implementation here.
                """
                reverse_nn = np.where(ndists <= self._radii[nixs])[0]
                reverse_nn = np.intersect1d(ixl, reverse_nn)
                Ws = len(reverse_nn) / nn
                """ Previous version (v0):

                    The weight of the supervised component is the fraction of
                    weights of the labeled instances in the total weight.

                    Ws = np.sum(w[ixl]) / np.sum(w)
                """

                # supervised score, new idea:
                if self.nn_sup == 'strict':
                    if len(reverse_nn) > 0:
                        ixa = np.where(labels[reverse_nn] > 0)[0]
                        Ss = np.sum(w[ixa]) / np.sum(w[reverse_nn])
                    else:
                        Ss = 0.0

            else:
                # supervised plays no role
                Ss = 0.0
                Ws = 0.0

            # combine supervised and unsupervised
            posterior[i] = (1.0 - Ws) * prior[i] + Ws * Ss

        # predictions
        offset = np.percentile(posterior, int(100 * (1.0 - self.c)))
        y_pred = np.ones(n, dtype=int)
        y_pred[posterior < offset] = -1

        return posterior, y_pred

    def _squashing_function(self, x, p):
        """ Compute the value of x under squashing function with parameter p. """
        return 1.0 - np.exp(np.log(0.5) * np.power(x / p, 2))
