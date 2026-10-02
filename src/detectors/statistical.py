"""Univariate statistical rules, built from scratch (numpy only).

Three rules live here:

* :class:`ThreeSigmaDetector` -- the field's default, and the cautionary
  protagonist. Uses the mean and standard deviation, both wrecked by the very
  outliers it is meant to catch (masking) and both assuming symmetry (swamping
  under skew).
* :class:`IQRDetector` -- the box-plot rule, built on quartiles, immune to
  masking. The robust drop-in replacement most analysts never adopt.
* :class:`ModifiedZScoreDetector` -- the median/MAD robust standard
  (Iglewicz & Hoaglin), included alongside IQR.

Each rule is natively univariate. Applied to multivariate ``X`` it flags a
point when *any* feature flags it (the standard column-wise OR), which is
exactly why these rules can only ever see global, per-feature anomalies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.detectors.base import as_2d

# 0.6745 == the 0.75 quantile of the standard normal; it scales the MAD so the
# modified z-score matches the ordinary z-score for normally distributed data.
_MAD_SCALE = 0.6745


class ThreeSigmaDetector:
    """Flag points more than ``k`` standard deviations from the mean."""

    name = "three_sigma"

    def __init__(self, k: float = 3.0) -> None:
        self.k = k

    def score_samples(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-point anomaly score: the largest |z| across features."""

        arr = as_2d(X)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=0)
        # Guard constant columns (std == 0) so they contribute no spurious score.
        safe_std = np.where(std == 0.0, 1.0, std)
        z = np.abs((arr - mean) / safe_std)
        z[:, std == 0.0] = 0.0
        return z.max(axis=1)

    def fit_predict(self, X: NDArray[np.float64]) -> NDArray[np.bool_]:
        return self.score_samples(X) > self.k


class IQRDetector:
    """Flag points beyond ``k`` * IQR from the quartiles (Tukey's rule)."""

    name = "iqr"

    def __init__(self, k: float = 1.5) -> None:
        self.k = k

    def _fences(self, col: NDArray[np.float64]) -> tuple[float, float, float]:
        q1, q3 = np.percentile(col, [25, 75])
        iqr = q3 - q1
        return q1 - self.k * iqr, q3 + self.k * iqr, iqr

    def score_samples(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """How far outside the fence a point is, in IQR units (0 if inside)."""

        arr = as_2d(X)
        scores = np.zeros(arr.shape[0])
        for j in range(arr.shape[1]):
            col = arr[:, j]
            lower, upper, iqr = self._fences(col)
            scale = iqr if iqr > 0 else 1.0
            below = np.maximum(lower - col, 0.0) / scale
            above = np.maximum(col - upper, 0.0) / scale
            scores = np.maximum(scores, np.maximum(below, above))
        return scores

    def fit_predict(self, X: NDArray[np.float64]) -> NDArray[np.bool_]:
        arr = as_2d(X)
        flagged = np.zeros(arr.shape[0], dtype=bool)
        for j in range(arr.shape[1]):
            col = arr[:, j]
            lower, upper, _ = self._fences(col)
            flagged |= (col < lower) | (col > upper)
        return flagged


class ModifiedZScoreDetector:
    """Robust modified z-score using the median and MAD."""

    name = "modified_zscore"

    def __init__(self, threshold: float = 3.5) -> None:
        self.threshold = threshold

    def score_samples(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-point score: the largest |modified z| across features."""

        arr = as_2d(X)
        median = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - median), axis=0)
        # When MAD is zero (>50% identical values) fall back to the mean abs
        # deviation so the score stays finite.
        mean_ad = np.mean(np.abs(arr - median), axis=0)
        scale = np.where(mad > 0, mad / _MAD_SCALE, mean_ad * 1.253314)
        safe_scale = np.where(scale == 0.0, 1.0, scale)
        mz = np.abs(arr - median) / safe_scale
        mz[:, scale == 0.0] = 0.0
        return mz.max(axis=1)

    def fit_predict(self, X: NDArray[np.float64]) -> NDArray[np.bool_]:
        return self.score_samples(X) > self.threshold
