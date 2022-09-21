"""
Anomaly score scalers. They map the anomaly scores to probabilities.

:author: Vincent Vercruyssen (2022)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import math
import numpy as np

from abc import abstractmethod, ABCMeta
from scipy.special import erf
from sklearn.base import BaseEstimator


# -------------------------------------
# base
# -------------------------------------


class BaseScaler(metaclass=ABCMeta):
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        # ! can change for different scalers!
        self.th = 0.5

    @abstractmethod
    def fit(self, S):
        pass

    @abstractmethod
    def _compute(self, S):
        pass

    def transform(self, S):
        probs = self._compute(S)
        return np.clip(probs, 0.0, 1.0)

    def get_threshold(self):
        return self.th

    def set_threshold(self, th):
        self.th = th


# -------------------------------------
# score scaling
# -------------------------------------


class NoneScaler(BaseEstimator, BaseScaler):
    def __init__(self, contamination=0.05):
        super().__init__(contamination=contamination)

    def fit(self, S):
        return self

    def _compute(self, S):
        return S


class LinearScaler(BaseEstimator, BaseScaler):
    def __init__(self, contamination=0.05):
        super().__init__(contamination=contamination)

    def fit(self, S):
        """Fit on training scores."""
        self.min_ = np.min(S)
        self.max_ = np.max(S)
        self.th = np.percentile(S, q=int((1.0 - self.contamination) * 100))
        self.th = (self.th - self.min_) / (self.max_ - self.min_)
        return self

    def _compute(self, S):
        """Transform test scores."""
        return (S - self.min_) / (self.max_ - self.min_)


class UnifyScaler(BaseEstimator, BaseScaler):
    def __init__(self, contamination=0.05):
        super().__init__(contamination=contamination)

    def fit(self, S):
        """Fit on training scores."""
        self.m_ = np.mean(S)
        self.s_ = np.std(S)
        return self

    def _compute(self, S):
        """Transform test scores."""
        # from: https://github.com/yzhao062/pyod/blob/master/pyod/models/base.py
        pre_erf_score = (S - self.m_) / (self.s_ * np.sqrt(2))
        erf_score = erf(pre_erf_score)
        return erf_score.clip(0, 1).ravel()


class SquashScaler(BaseEstimator, BaseScaler):
    def __init__(self, contamination=0.05):
        super().__init__(contamination=contamination)

    def fit(self, S):
        """Fit on training scores."""
        self.t_ = np.percentile(S, q=int((1.0 - self.contamination) * 100))
        return self

    def _compute(self, S):
        """Transform test scores."""
        return 1.0 - np.exp(np.log(0.5) * np.power(S / self.t_, 2))


class SigmoidScaler(BaseEstimator, BaseScaler):
    def __init__(self, contamination=0.05):
        super().__init__(contamination=contamination)

    def fit(self, S):
        """Fit on training scores."""
        self.anchor_ = np.percentile(S, q=int((1.0 - self.contamination) * 100))
        self.spread_ = abs(self.anchor_ - np.median(S))
        return self

    def _compute(self, S):
        """Transform test scores."""
        spread_target = 0.05

        # small spread
        if self.spread_ is None or abs(self.spread_) < 1e-5:
            return 1.0 / (1.0 + np.exp(-S + self.anchor_))

        # anchor and spread
        t = (1.0 - spread_target) / spread_target
        v1 = self.anchor_  # f(v1) = 0.5
        v2 = self.anchor_ - self.spread_  # f(v2) = t
        w = math.log(t) / (v1 - v2)
        g = w * v1
        return 1.0 / (1.0 + np.exp(-w * S + g))
