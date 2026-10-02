"""Local Outlier Factor: the detector built for *local* anomalies.

LOF compares a point's local density to that of its neighbours. A point in a
sparse pocket surrounded by dense neighbourhoods is flagged even when it sits
comfortably inside the global range -- the local outlier that global rules
(three-sigma, IQR) and single-cloud methods cannot see.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import LocalOutlierFactor

from src.detectors.base import as_2d


class LOFDetector:
    """sklearn Local Outlier Factor with the shared detector interface."""

    name = "lof"

    def __init__(self, contamination: float = 0.08, n_neighbors: int = 20) -> None:
        self.contamination = contamination
        # n_neighbors cannot exceed n_samples - 1; clamped at fit time.
        self.n_neighbors = n_neighbors

    def _model(self, n_samples: int) -> LocalOutlierFactor:
        k = max(1, min(self.n_neighbors, n_samples - 1))
        return LocalOutlierFactor(n_neighbors=k, contamination=self.contamination)

    def score_samples(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = as_2d(X)
        model = self._model(arr.shape[0])
        model.fit_predict(arr)
        # negative_outlier_factor_: more negative == more anomalous; negate so
        # higher == more anomalous, matching the shared convention.
        return -model.negative_outlier_factor_

    def fit_predict(self, X: NDArray[np.float64]) -> NDArray[np.bool_]:
        arr = as_2d(X)
        model = self._model(arr.shape[0])
        return model.fit_predict(arr) == -1
