""" iNNe based anomaly detection.

Reference:
    T. R. Bandaragoda, K. Ming Ting, D. Albrecht, F. T. Liu, Y. Zhu, and J. R. Wells.
    Isolation-based anomaly detection using nearest-neighbor ensembles.
    In Computational Intelligence, vol. 34, 2018, pp. 968-998.

"""

# Authors: Vincent Vercruyssen, 2018.

import math
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric

from .BaseDetector import BaseDetector
from ..utils.validation import check_X_y


# -------------
# MAIN CLASS
# -------------

class iNNe(BaseDetector):
    """ Isolation-based nearest neighbor ensemble.

    Parameters
    ----------
    t : int (default=100)
        Number of estimators to construct.

    n : int (default=16)
        Number of examples in each subsample.

    contamination : float (default=0.1)
        Estimate of the expected percentage of anomalies in the data.

    metric : string (default=euclidean)
        Distance metric for constructing the BallTree: euclidean, cosine

    Comments
    --------
    - The number of examples in each subsample cannot be larger than the number of instances in
    the data: automatically correct if necessary.
    """

    def __init__(self, t=100, n=16, contamination=0.1, metric='euclidean',
                 tol=1e-8, verbose=False):
        super(iNNe, self).__init__()

        self.t = int(t)
        self.n = int(n)
        self.contamination = float(contamination)
        if metric == 'cosine':
            self.metric = cosine_similarity
        else:
            try:
                self.metric = DistanceMetric.get_metric(metric)
            except ValueError as e:
                raise BaseException(e)

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

        self.n = min(self.n, n)

        # construct the ensembles with random sampling of the points
        self.ensemble = []
        for _ in range(self.t):
            # 1. randomly sample a number of points
            sample_ix = np.random.choice(n, self.n, replace=False)
            subsample = X[sample_ix, :]

            # 2. construct the set of hyperspheres (single member of the ensemble)
            member = hyperSpheres(self.n, metric=self.metric)
            member.construct_hyperspheres(sample_ix, subsample)

            # 3. add to the ensemble
            self.ensemble.append(member)

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

        # compute the anomaly score using each member of the ensemble
        y_score = np.zeros(n, dtype=float)
        for i, member in enumerate(self.ensemble):
            i_scores = member.compute_isolation_scores(X)
            y_score += i_scores
        i_scores = i_scores / self.t

        y_score = (i_scores - min(i_scores)) / (max(i_scores) - min(i_scores))

        # prediction threshold + absolute predictions
        self.threshold = np.sort(y_score)[int(n * (1.0 - self.contamination))]
        y_pred = np.ones(n, dtype=float)
        y_pred[y_score < self.threshold] = -1

        return y_score, y_pred



# -------------
# SINGLE MEMBER
# -------------

class hyperSpheres:
    """ Single member of the ensemble """

    def __init__(self, n, metric):

        self.metric = metric  # this metric has a function to compute the distance
        self.spheres = np.zeros((3, n), dtype=float)

    def construct_hyperspheres(self, sample_ix, subsample):
        """ Construct the hyperspheres. """

        # 1. compute pairwise distance matrix
        Dmat = self.metric.pairwise(subsample)
        np.fill_diagonal(Dmat, np.inf)  # a point cannot find itself as the nearest neighbor

        # 2. find nearest neighbor and distance to nearest neighbor
        nn_dist = np.min(Dmat, axis=0)
        nn_ix = np.argmin(Dmat, axis=0)

        # 3. compute isolation score of each hypersphere
        scores = 1.0 - (nn_dist[nn_ix] / nn_dist)
        scores = np.nan_to_num(scores)

        # 4. store info
        self.spheres[0, :] = sample_ix      # index of the center
        self.spheres[1, :] = nn_dist        # hypersphere radius
        self.spheres[2, :] = scores         # isolation scores

        return self

    def compute_isolation_scores(self, data):
        """ Compute the isolation score for each point in the data. """

        n = len(data)
        i_scores = np.zeros(n, dtype=float)

        # 1. find the center points of the hyperspheres
        ix_c = self.spheres[0, :].astype(int)
        centers = data[ix_c, :]

        # 2. compute the scores using chunks (to reduce memory usage)
        i = 0
        chunk_size = 1000
        end_reached = False
        while not end_reached:
            # select the data chunk
            data_chunk = data[i:i+chunk_size, :]

            # compute the distance matrix
            Dmat = self.metric.pairwise(data_chunk, centers)

            # find the closest center and check if x in hypersphere
            nn_dist = np.min(Dmat, axis=1)
            nn_ix = np.argmin(Dmat, axis=1)
            radii = self.spheres[1, nn_ix]
            scores = self.spheres[2, nn_ix]

            # fill in the score vector
            for j in range(len(data_chunk)):
                if nn_dist[j] < radii[j]:
                    i_scores[i+j] = scores[j]
                else:
                    i_scores[i+j] = 1.0

            # stopping criterion + increasing index with chunksize
            i += chunk_size
            if i > n:
                end_reached = True
                break

        return i_scores
