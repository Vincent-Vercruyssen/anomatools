# -*- coding: UTF-8 -*-
"""

Plotting functions.

:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from sklearn.utils.validation import check_X_y

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('Import ERROR: `matplotlib` should be installed for this file!')


# ----------------------------------------------------------------------------
# functions
# ----------------------------------------------------------------------------

def plot_2D_normals_anomalies(X, y, figure_size=(6, 6)):
    """ Plot dataset of 2D normals and anomalies. """

    # markers and colors
    marker_style = {1.0: 'o', -1.0: 'x'}
    color_style = {1.0: 'red', -1.0: 'blue'}
    
    # make the figure
    plt.figure(figsize=figure_size)
    for i, x in enumerate(X):
        plt.plot(x[0], x[1], marker_style[y[i]], color=color_style[y[i]])
    plt.grid(alpha=0.5)
    plt.show()


def plot_2D_classifier(clf, X, y=None, steps=50, margin_size=0.1, predict_scores=True, contours=True, figure_size=(6, 6)):
    """ Plot 2D contours of a classifier.
        Fits and predicts the classifier itself.
    """

    # check input
    if y is None:
        X, _ = check_X_y(X, np.zeros(len(X)))
    else:
        X, y = check_X_y(X, y)

    # ranges
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)

    xmin, xmax = x_min - margin_size * x_range, x_max + margin_size * x_range
    ymin, ymax = y_min - margin_size * y_range, y_max + margin_size * y_range

    # make the meshgrid based on the data 
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, int(steps)),
        np.linspace(ymin, ymax, int(steps)))
    X_mesh = np.c_[xx.ravel(), yy.ravel()]

    # fit and predict the classifier
    if y is None:
        clf.fit(X)
    else:
        clf.fit(X, y)
    if predict_scores:
        Z = clf.predict_proba(X_mesh)[:, 1]
    else:
        Z = clf.predict(X_mesh)
    Z = Z.reshape(xx.shape)

    # make the figure
    plt.figure(figsize=figure_size)

    # plot the contour
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.colorbar()
    if contours:
        plt.contour(xx, yy, Z, np.linspace(0.0, np.max(Z), 10))

    # plot the points
    plt.scatter(X[:, 0], X[:, 1], cmap=plt.cm.coolwarm, s=40, edgecolors='k')
    plt.show()
