"""
Validation functions.

:author: Vincent Vercruyssen (2022)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np

from inspect import isclass
from sklearn.utils import check_X_y


# -------------------------------------
# functions
# -------------------------------------


def check_is_fitted(estimator, fitted, attributes=None):
    """Check whether the estimator is fitted.

    Parameters
    ----------
    estimator : object (class instance)
        A fitted estimator.

    fitted : bool
        The fitted attribute of the estimator.

    attributes : list, optional
        List of strings, e.g., ['threshold\_', 'coef\_'].
        These are the essential parameters resulting from a fit.
        If None, the function only looks at fitted.

    Raises
    ------
    Exception
        The detector is not fitted.
    """

    if isclass(estimator):
        raise TypeError(f"ERROR: `{estimator}` is a class, not an instance")

    if not hasattr(estimator, "fit"):
        raise TypeError(f"ERROR: each estimator should implement a fit() function")

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        attrs = all([hasattr(estimator, attr) for attr in attributes])
    else:
        # estimator does not have to have attributes
        attrs = True

    if not fitted or not attrs:
        raise Exception(f"ERROR: the estimator is not fitted")


def check_arrays_X_y(X, y=None, *args, **kwargs):
    """Check the input data for the classifier.

    Wrapper function around Scikit-learn's ``check_X_y`` function.
    Can deal with y=None, which happens often in unsupervised learning.

    Parameters
    ----------
    X : np.ndarray
        The input instances.

    y : np.ndarray, optional
        The input instance class labels.

    Returns
    -------
    Xc : object
        The converted and validated X.

    yc : object
        The converted and validated y.

    Raises
    ------
    Exception
        The input data are not valid.
    """

    # check y
    if y is None:
        y = np.zeros(len(X))

    # sklearn check
    try:
        return check_X_y(X, y, dtype=None, *args, **kwargs)
    except Exception as e:
        raise Exception(f"ERROR: {e}")
