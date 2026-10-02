"""Isolation Forest: the modern default for tabular anomaly detection.

Anomalies are *few and different*, so a point that random axis-aligned cuts
isolate in only a handful of splits is likely anomalous. This module ships
both the sklearn detector (used in the arena) and a from-scratch
implementation of the core intuition -- how many random cuts it takes to
isolate a point -- whose ranking ``test_detectors.py`` checks against sklearn.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest

from src.detectors.base import as_2d


def _avg_unsuccessful_path(n: int) -> float:
    """Average path length of an unsuccessful BST search over ``n`` points.

    This is ``c(n)`` from the Isolation Forest paper, used to normalise path
    lengths so scores are comparable across subsample sizes.
    """

    if n <= 1:
        return 0.0
    harmonic = np.log(n - 1) + np.euler_gamma
    return 2.0 * harmonic - 2.0 * (n - 1) / n


def _path_length(
    x: NDArray[np.float64],
    data: NDArray[np.float64],
    rng: np.random.Generator,
    depth: int,
    max_depth: int,
) -> float:
    """Depth at which a single random isolation tree isolates ``x``."""

    n = data.shape[0]
    if n <= 1 or depth >= max_depth:
        return depth + _avg_unsuccessful_path(n)
    # Choose a feature with non-zero spread, then a random split within it.
    spreads = data.max(axis=0) - data.min(axis=0)
    usable = np.flatnonzero(spreads > 0)
    if usable.size == 0:
        return depth + _avg_unsuccessful_path(n)
    feature = int(rng.choice(usable))
    column = data[:, feature]
    lo, hi = float(np.min(column)), float(np.max(column))
    split = rng.uniform(lo, hi)
    mask = data[:, feature] < split
    branch = data[mask] if x[feature] < split else data[~mask]
    return _path_length(x, branch, rng, depth + 1, max_depth)


def isolation_depth_scores(
    X: NDArray[np.float64],
    n_trees: int = 100,
    subsample: int = 256,
    seed: int = 0,
) -> NDArray[np.float64]:
    """From-scratch Isolation Forest anomaly scores (higher == more anomalous).

    Returns ``s = 2 ** (-E[h(x)] / c(subsample))`` averaged over ``n_trees``
    random trees, matching the scoring convention of the original paper.
    """

    arr = as_2d(X)
    n = arr.shape[0]
    size = min(subsample, n)
    max_depth = int(np.ceil(np.log2(max(size, 2))))
    rng = np.random.default_rng(seed)
    totals = np.zeros(n)
    for _ in range(n_trees):
        sample = arr[rng.choice(n, size=size, replace=False)]
        for i in range(n):
            totals[i] += _path_length(arr[i], sample, rng, 0, max_depth)
    expected = totals / n_trees
    return 2.0 ** (-expected / _avg_unsuccessful_path(size))


class IsolationForestDetector:
    """sklearn Isolation Forest with the shared detector interface."""

    name = "isolation_forest"

    def __init__(
        self, contamination: float = 0.08, n_estimators: int = 200, seed: int = 0
    ) -> None:
        self.model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination, random_state=seed
        )

    def score_samples(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        # sklearn score_samples: higher == more normal, so negate for our convention.
        arr = as_2d(X)
        return -self.model.fit(arr).score_samples(arr)

    def fit_predict(self, X: NDArray[np.float64]) -> NDArray[np.bool_]:
        arr = as_2d(X)
        return self.model.fit_predict(arr) == -1
