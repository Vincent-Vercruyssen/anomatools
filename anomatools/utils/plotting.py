# -*- coding: UTF-8 -*-
"""

Plotting functions.

:author: Vincent Vercruyssen
:year: 2018
:license: Apache License, Version 2.0, see LICENSE for details.

"""

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
    plt.show()


def plot_2D_classifier(clf, X, y, h=)

 
def plot_classifier_contour_2D(clf, X, y=np.array([]), xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, h=0.02, auto_range=False):
    """ Plot a contour plot for classifier """
    
    # make the meshgrid
    if auto_range:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    else:
        x_min, x_max = xmin, xmax
        y_min, y_max = ymin, ymax
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Xtest = np.c_[xx.ravel(), yy.ravel()]
    
    # fit and predict
    if len(y) == 0:
        clf.fit(X)
    else:
        clf.fit(X, y)
    Z, _ = clf.predict(Xtest)
    Z = Z.reshape(xx.shape)
    
    # plot the heatmap and the normal points
    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], cmap=plt.cm.coolwarm, s=40, edgecolors='k')
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.colorbar()
    plt.contour(xx, yy, Z, np.linspace(0.0, np.max(Z), 10))
    plt.show()
