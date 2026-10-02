"""Mahalanobis-distance detector, built from scratch (numpy/scipy only).

The Mahalanobis distance measures how far a point is from the center *after*
accounting for the covariance of the data, so it finds multivariate outliers
that are unremarkable on every single feature but violate the correlation
structure of the cloud -- the per-feature-normal, combination-abnormal point
the univariate rules cannot see.

The squared distances match :class:`sklearn.covariance.EmpiricalCovariance`
(both use the maximum-likelihood covariance, divided by ``n``); this is what
``test_detectors.py`` asserts.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from src.detectors.base import as_2d


class MahalanobisDetector:
    """Flag points with a large covariance-adjusted distance from the center."""

    name = "mahalanobis"

    def __init__(self, contamination: float = 0.08, threshold: float | None = None) -> None:
        self.contamination = contamination
        # If given, a chi-square tail probability (e.g. 0.975) is used instead
        # of the empirical contamination quantile.
        self.chi2_confidence = threshold
        self._mean: NDArray[np.float64] | None = None
        self._precision: NDArray[np.float64] | None = None

    def _fit(self, arr: NDArray[np.float64]) -> None:
        self._mean = arr.mean(axis=0)
        # bias=True -> divide by n, matching sklearn's EmpiricalCovariance (MLE).
        cov = np.cov(arr, rowvar=False, bias=True)
        cov = np.atleast_2d(cov)
        self._precision = np.linalg.pinv(cov)

    def mahalanobis_sq(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Squared Mahalanobis distance of every row of ``X``."""

        arr = as_2d(X)
        if self._mean is None or self._precision is None:
            self._fit(arr)
        assert self._mean is not None and self._precision is not None
        centered = arr - self._mean
        return np.einsum("ij,jk,ik->i", centered, self._precision, centered)

    def score_samples(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.mahalanobis_sq(X)

    def fit_predict(self, X: NDArray[np.float64]) -> NDArray[np.bool_]:
        arr = as_2d(X)
        d2 = self.mahalanobis_sq(arr)
        if self.chi2_confidence is not None:
            cutoff = chi2.ppf(self.chi2_confidence, df=arr.shape[1])
        else:
            cutoff = np.quantile(d2, 1.0 - self.contamination)
        return d2 > cutoff
