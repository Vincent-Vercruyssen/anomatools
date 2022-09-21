"""
Test functionality.
`
:author: Vincent Vercruyssen (2022)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from anomatools.utils.utils import DistanceFun
from anomatools.models import BaseDetector
from anomatools.models import kNNo
from anomatools.models import iNNE
from anomatools.models import ClusterBasedAD
from anomatools.models import SSDO
from anomatools.models import SSkNNO


from sklearn.metrics import pairwise_distances


# -------------------------------------
# vars
# -------------------------------------

PLOT_COLORS = {1: "tab:red", -1: "tab:green", 0: "tab:blue"}
PLOT_MARKERS = {1: "^", -1: "*", 0: "o"}


# -------------------------------------
# func
# -------------------------------------


def run_test():
    # test distance computations
    # x = np.array([1, 0.2, 3])
    # y = np.array([4, 22, 3])
    # X = np.random.rand(100, 3)
    # Y = np.random.rand(100, 3)

    # d1 = DistanceFun().pairwise_single(x, y)
    # d2 = DistanceFun().pairwise_multiple(X, Y)
    # d2o = pairwise_distances(X, Y)
    # assert np.allclose(d2, d2o)

    # distcalc = DistanceFun().fit(X)
    # D, I = distcalc.search_neighbors(Y, 3)
    # D, I = distcalc.search_radius(Y, 0.5)
    # D, I = distcalc.search_neighbors(X, 3)
    # D, I = distcalc.search_radius(X, 0.5)
    # print("\nDistance checks passed")

    # test models
    # mdl = kNNo().fit(X)
    # labels = mdl.predict(Y)
    # scores = mdl.predict_proba(Y)
    # print("\nKNNO\n", labels.shape, scores.shape)

    # mdl = iNNE().fit(X)
    # labels = mdl.predict(Y)
    # scores = mdl.predict_proba(Y)
    # print("\nINNE\n", labels.shape, scores.shape)

    # mdl = ClusterBasedAD(base_clusterer=KMeans(n_clusters=14)).fit(X)
    # labels = mdl.predict(Y)
    # scores = mdl.predict_proba(Y)
    # print("\nClusterBasedAD\n", labels.shape, scores.shape)

    # mdl = SSDO().fit(X)
    # labels = mdl.predict(Y)
    # scores = mdl.predict_proba(Y)
    # print("\nSSDO\n", labels.shape, scores.shape)

    # mdl = SSkNNO().fit(X)
    # labels = mdl.predict(Y)
    # scores = mdl.predict_proba(Y)
    # print("\nSSkNNO\n", labels.shape, scores.shape)

    # l = np.zeros(len(X), dtype=float)
    # l[:5] = 1.0
    # l[-10:] = -1.0
    # mdl = SSkNNO().fit(X, l)
    # labels = mdl.predict(Y)
    # scores = mdl.predict_proba(Y)
    # print("\nSSkNNO\n", labels.shape, scores.shape)

    # l = np.zeros(len(X), dtype=float)
    # l[:5] = 1.0
    # l[-10:] = -1.0
    # mdl = SSDO().fit(X, l)
    # labels = mdl.predict(Y)
    # scores = mdl.predict_proba(Y)

    # fit and plot the decision boundary
    # print("\nUNSUPERVISED: kNNo")
    # X, y = generate_2D_dataset()
    # mdl = kNNo().fit(X)
    # plot_2D_decision_boundary(X, y=y, trained_classifier=mdl, title="kNNO plot")

    # print("\nUNSUPERVISED: iNNE")
    # X, y = generate_2D_dataset()
    # mdl = iNNE().fit(X)
    # plot_2D_decision_boundary(X, y=y, trained_classifier=mdl, title="iNNE plot")

    # print("\nUNSUPERVISED: ClusterBasedAD")
    # X, y = generate_2D_dataset()
    # mdl = ClusterBasedAD(base_clusterer=KMeans(n_clusters=6)).fit(X)
    # plot_2D_decision_boundary(
    #     X, y=y, trained_classifier=mdl, title="ClusterBasedAD plot"
    # )

    # print("\nSEMI-SUPERVISED: SSkNNO")
    # X, y = generate_2D_dataset()
    # mdl = SSkNNO().fit(X, y)
    # plot_2D_decision_boundary(X, y=y, trained_classifier=mdl, title="SSkNNO plot")

    print("\nSEMI-SUPERVISED: SSDO")
    X, y = generate_2D_dataset()
    mdl = SSDO().fit(X, y)
    plot_2D_decision_boundary(X, y=y, trained_classifier=mdl, title="SSDO plot")


def plot_2D_decision_boundary(
    X,
    y=None,
    trained_classifier=None,
    margin_size=0.1,
    steps=100,
    figsize=(13, 10),
    title="",
):
    """Plot the decision boundary of the classifier in 2D."""

    # range
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)

    xmin, xmax = x_min - margin_size * x_range, x_max + margin_size * x_range
    ymin, ymax = y_min - margin_size * y_range, y_max + margin_size * y_range

    # plot points
    plt.figure(figsize=figsize)
    if y is not None:
        for l in [0, -1, 1]:
            ixl = np.where(y == l)[0]
            plt.scatter(
                X[ixl, 0], X[ixl, 1], color=PLOT_COLORS[l], marker=PLOT_MARKERS[l]
            )
    else:
        plt.scatter(X[:, 0], X[:, 1], edgecolors="k")

    # plot decision boundary
    if trained_classifier is not None:
        xx, yy = np.meshgrid(
            np.linspace(xmin, xmax, int(steps)), np.linspace(ymin, ymax, int(steps))
        )
        X_mesh = np.c_[xx.ravel(), yy.ravel()]

        # predict for the mesh
        Z = trained_classifier.predict_proba(X_mesh)[:, 1]  # anom prob
        Z = Z.reshape(xx.shape)

        # plot figure
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, vmin=0.0, vmax=1.0, alpha=0.8)
        plt.colorbar()

    # clean plot
    plt.grid(alpha=0.25)
    plt.title(title)
    plt.show()


def generate_2D_dataset(seed=1, contamination=0.05):
    """Generate 2D toy dataset."""

    np.random.seed(seed)

    # generate X
    n1 = 200
    n2 = 400
    n3 = 500
    n4 = 50
    nn = n1 + n2 + n3 + n4
    na = int(nn * contamination)
    X = np.vstack(
        (
            np.random.uniform(low=[2, 12], high=[8, 18], size=(n1, 2)),
            np.random.multivariate_normal([7, 5], [[0.2, 0], [0, 0.2]], n2),
            np.random.multivariate_normal([10, 11], [[0.3, 0.4], [0.4, 3]], n3),
            np.random.multivariate_normal([12, 8], [[0.05, 0], [0, 0.05]], n4),
            np.random.uniform(low=[0, 0], high=[15, 20], size=(na, 2)),
        )
    )

    # y: when the label of an example = 0, it means we do not know its label
    y = np.zeros(len(X), dtype=int)

    # let's assume we know 50 normal examples... (label = -1)
    rand_ix = np.random.choice(nn, 50, replace=False)
    y[rand_ix] = -1

    # ... and 5 anomalous examples (label = 1)
    y[-5:] = 1

    return X, y


if __name__ == "__main__":
    run_test()
