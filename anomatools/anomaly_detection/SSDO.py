""" Semi-Supervised Detection of Anomalies """

# Authors: Vincent Vercruyssen

import numpy as np
import scipy.stats as sps

from collections import Counter
from scipy.spatial import cKDTree
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .BaseDetector import BaseDetector
from ..clustering.COPKMeans import COPKMeans
from ..utils.fastfuncs import fast_distance_matrix
from ..utils.validation import check_X_y


class SSDO(BaseDetector):
    """ Semi-Supervised Detection of Anomalies (SSDO)

    Parameters
    ----------
    n_clusters : int (default=10)
        Number of clusters used for the COP k-means clustering algorithm.

    alpha : float (default=2.3)
        User influence parameter that controls the weight given to the
        unsupervised and label propragation components of an instance's
        anomaly score.

    k : int (default=30)
        Controls how many instances are updated by propagating the label of a
        single labeled instance.

    contamination : float (default=0.1)
        Estimate of the expected percentage of anomalies in the data.

    base_classifier : str (default='ssdo')
        Unsupervised baseline classifier:
            'ssdo'    --> SSDO baseline (based on constrained k-means clustering)
            'IF'      --> IsolationForest as the base classifier
            'LOF'     --> Local Outlier Factor as the base classifier
            'other'   --> use a different classifier passed to SSDO
    """

    def __init__(self,
            n_clusters=10,                      # number of clusters for the SSDO base classifier
            alpha=2.3,                          # the alpha parameter for SSDO label propagation
            k=30,                               # the k parameter for SSDO label propagation
            contamination=0.1,                  # expected number of anomalies in the data
            base_classifier='ssdo',             # type of base classifier to use
            base_classifier_parameters={},      # parameters for the base classifier if applicable
            tol=1e-8,                           # tolerance level
            verbose=False):
        super(BaseDetector, self).__init__()

        # instantiate the parameters
        self.nc = int(n_clusters)
        self.alpha = float(alpha)
        self.k = int(k)
        self.c = float(contamination)
        self.base_classifier = str(base_classifier)
        self.base_classifier_parameters = base_classifier_parameters
        self.tol = tol
        self.verbose = bool(verbose)

    def fit_predict(self, X, y=None, base_classifier=None):
        """ Fit the model to the training set X and returns the anomaly score
            of the instances in X.

        :param X : np.array(), shape (n_samples, n_features)
            The samples to compute anomaly score w.r.t. the training samples.
        :param y : np.array(), shape (n_samples), default = None
            Labels for examples in X.
        :param base_classifier : object
            Base classifier to detect the anomalies if SSDO is not used.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        return self.fit(X, y, base_classifier).predict(X)

    def fit(self, X, y=None, base_classifier=None):
        """ Fit the model using data in X.

        :param X : np.array(), shape (n_samples, n_features)
            The samples to compute anomaly score w.r.t. the training samples.
        :param y : np.array(), shape (n_samples), default = None
            Labels for examples in X.
        :param base_classifier : object
            Base classifier to detect the anomalies if SSDO is not used.
        :param base_classifier_parameters : dictionary


        :returns self : object
        """

        X, y = check_X_y(X, y)
        n, _ = X.shape
        if y is None:
            y = np.zeros(n, dtype=int)

        # compute the prior using different base classifiers
        if self.base_classifier == 'ssdo':
            # COPKMeans classifier
            self._fit_prior_parameters(X, y)
        elif self.base_classifier == 'other':
            # check the validity of the classifier
            if not isinstance(base_classifier, object):
                raise ValueError('ERROR: `base_classifier` should be an object with a fit() and predict()')
            has_fit = callable(getattr(base_classifier, 'fit', None))
            has_pre = callable(getattr(base_classifier, 'predict', None))
            if not has_fit:
                raise Exception('ERROR: `base_classifier` has not fit() function')
            if not has_pre:
                raise Exception('ERROR: `base_classifier` has no predict() function')
            self.prior_detector = base_classifier
            self.prior_detector.fit(X)
        elif self.base_classifier == 'LOF':
            # use Local Oultier Factor
            self.prior_detector = LocalOutlierFactor(**self.base_classifier_parameters, novelty=True)
            self.prior_detector.fit(X)
        elif self.base_classifier == 'IF':
            # use Isolation Forest
            self.prior_detector = IsolationForest(n_estimators=500, max_samples='auto')
            self.prior_detector.fit(X)
        else:
            raise ValueError('ERROR: invalid choice of `base_classifier` (`ssdo`, `IF`, or `other`)')

        # compute eta parameter
        self.eta = self._compute_eta(X)

        # store the labeled points
        self.labels_available = False
        ixl = np.where(y != 0.0)[0]
        if len(ixl) > 0:
            self.labels_available = True
            self._labels = y[ixl]
            self._X_labels = X[ixl, :]

        return self

    def predict(self, X):
        """ Compute the anomaly score for unseen instances.

        :param X : np.array(), shape (n_samples, n_features)
            The samples in the test set for which to compute the anomaly score.

        :returns y_score : np.array(), shape (n_samples)
            Anomaly score for the examples in X.
        :returns y_pred : np.array(), shape (n_samples)
            Returns -1 for inliers and +1 for anomalies/outliers.
        """

        X, _ = check_X_y(X, None)
        n, _ = X.shape

        # compute the prior
        if self.base_classifier == 'ssdo':
            prior = self._compute_prior(X)
            # already normalized between 0 and 1
        elif self.base_classifier == 'other':
            print('WARNING: SSDO expects two outputs from the predict() function of the classifier: \
                    1) the y_prob, and 2) the y_pred. It will use the y_prob as the prior.')
            prior, _ = self.prior_detector.predict(X)
            prior = (prior - min(prior)) / (max(prior) - min(prior))
        elif self.base_classifier == 'LOF':
            # inverse of the scores: higher is more anomalous
            prior = self.prior_detector.score_samples(X) * -1
            # rescale between 0 and 1
            prior = (prior - min(prior)) / (max(prior) - min(prior))
        elif self.base_classifier == 'IF':
            prior = self.prior_detector.decision_function(X) * -1
            prior = (prior - min(prior)) / (max(prior) - min(prior))
        else:
            print('WARNING: no `base_classifier` for predict()')
            prior = np.ones(n, dtype=np.float)

        # scale the prior using the squashing function
        # TODO: this is the expected contamination in the test set!
        gamma = np.sort(prior)[int(n * (1.0 - self.c))] + self.tol
        prior = np.array([1 - self._squashing_function(x, gamma) for x in prior])

        # compute the posterior
        if self.labels_available:
            y_score = self._compute_posterior(X, prior, self.eta)
        else:
            y_score = prior

        # y_pred (using the expected contamination)
        # TODO: this is the expected contamination in the test set!
        offset = np.sort(y_score)[int(n * (1.0 - self.c))]
        y_pred = np.ones(n, dtype=int)
        y_pred[y_score < offset] = -1

        return y_score, y_pred

    def _fit_prior_parameters(self, X, y):
        """ Fit the parameters for computing the prior score:
            - (constrained) clustering
            - cluster size
            - max intra-cluster distance
            - cluster deviation
        """

        # construct cannot-link constraints + remove impossible cannot-links
        ixn = np.where(y == -1.0)[0]
        ixa = np.where(y == 1.0)[0]
        cl = np.array(np.meshgrid(ixa, ixn)).T.reshape(-1,2)

        # cluster
        self.clus = COPKMeans(n_clusters=self.nc)
        centroids, labels = self.clus.fit_predict(X, cannot_link=cl)
        self.nc = self.clus.n_clusters

        # cluster sizes (Counter sorts by key!)
        self.cluster_sizes = np.array(list(Counter(labels).values())) / max(Counter(labels).values())

        # compute the max intra-cluster distance
        self.max_intra_cluster = np.zeros(self.nc, dtype=np.float)
        for i, l in enumerate(labels):
            c = centroids[l, :]
            d = np.linalg.norm(X[i, :] - c)
            if d > self.max_intra_cluster[l]:
                self.max_intra_cluster[l] = d

        # compute the inter-cluster distances
        if self.nc == 1:
            self.cluster_deviation = np.array([1])
        else:
            inter_cluster = np.ones(self.nc, dtype=np.float) * np.inf
            for i in range(self.nc):
                for j in range(self.nc):
                    if i != j:
                        d = np.linalg.norm(centroids[i, :] - centroids[j, :])
                        if not(d < self.tol) and d < inter_cluster[i]:
                            inter_cluster[i] = d
        self.cluster_deviation = inter_cluster / max(inter_cluster)

        return self

    def _compute_prior(self, X):
        """ Compute the constrained-clustering-based outlier score.

        :returns prior : np.array(), shape (n_samples)
            Prior anomaly score between 0 and 1.
        """

        n, _ = X.shape

        # predict the cluster labels + distances to the clusters
        _, labels, distances = self.clus.predict(X, include_distances=True)

        # compute the prior
        prior = np.zeros(n)
        for i, l in enumerate(labels):
            if self.max_intra_cluster[l] < self.tol:
                point_deviation = 1.0
            else:
                point_deviation = distances[i] / self.max_intra_cluster[l]
            prior[i] = (point_deviation * self.cluster_deviation[l]) / self.cluster_sizes[l]

        return prior

    def _compute_posterior(self, X, prior, eta):
        """ Update the clustering score with label propagation.

        :returns posterior : np.array(), shape (n_samples)
            Posterior anomaly score between 0 and 1.
        """

        n, _ = X.shape

        # labeled examples
        ixa = np.where(self._labels == 1.0)[0]
        ixn = np.where(self._labels == -1.0)[0]

        # compute limited distance matrices (to normals, to anomalies)
        Dnorm = fast_distance_matrix(X, self._X_labels[ixn, :])
        Danom = fast_distance_matrix(X, self._X_labels[ixa, :])

        # compute posterior
        posterior = np.zeros(n)
        for i in range(n):
            # weighted distance to anomalies & normals
            da = np.sum(self._squashing_function(Danom[i, :], eta))
            dn = np.sum(self._squashing_function(Dnorm[i, :], eta))
            # posterior
            z = 1.0 / (1.0 + self.alpha * (da + dn))
            posterior[i] = z * (prior[i] + self.alpha * da)

        return posterior

    def _compute_eta(self, X):
        """ Compute the eta parameter.

        :returns eta : float
            Eta parameter is the harmonic mean of the k-distances.
        """

        n, _ = X.shape

        # construct KD-tree
        tree = cKDTree(X, leafsize=16)

        # query distance to k'th nearest neighbor of each point
        d = np.zeros(n)
        for i, x in enumerate(X):
            dist, _ = tree.query(x, k=self.k+1)
            d[i] = dist[-1]

        # compute eta as the harmonic mean of the k-distances
        filler = min(d[d > 0.0])
        d[d == 0.0] = filler
        eta = sps.hmean(d)

        if eta < self.tol:
            eta = self.tol

        return eta

    def _squashing_function(self, x, p):
        """ Compute the value of x under squashing function with parameter p. """
        return np.exp(np.log(0.5) * np.power(x / p, 2))
