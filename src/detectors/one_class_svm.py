"""One-class SVM: the kernel-based boundary detector.

Learns a boundary around the bulk of the (standardised) data and flags
anything outside it. Included for completeness and for its distinct failure
profile -- it is sensitive to scaling and to the ``nu`` setting, and does not
assume the single-Gaussian cloud Mahalanobis does.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.svm import OneClassSVM

from src.detectors.base import as_2d


class OneClassSVMDetector:
    """sklearn One-class SVM on standardised features."""

    name = "one_class_svm"

    def __init__(self, contamination: float = 0.08, gamma: str | float = "scale") -> None:
        # nu upper-bounds the fraction of training points allowed outside the
        # boundary, so the contamination rate is its natural setting.
        self.model = OneClassSVM(nu=max(min(contamination, 1.0), 1e-3), gamma=gamma)

    def _standardise(self, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=0)
        return (arr - mean) / np.where(std == 0.0, 1.0, std)

    def score_samples(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = self._standardise(as_2d(X))
        # decision_function: higher == more normal, so negate.
        return -self.model.fit(arr).decision_function(arr)

    def fit_predict(self, X: NDArray[np.float64]) -> NDArray[np.bool_]:
        arr = self._standardise(as_2d(X))
        return self.model.fit_predict(arr) == -1
