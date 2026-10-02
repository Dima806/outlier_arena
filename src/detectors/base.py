"""Shared detector protocol and helpers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Detector(Protocol):
    """The uniform interface every detector implements.

    ``fit_predict`` returns a boolean mask (``True`` == outlier).
    ``score_samples`` returns a float score where higher == more anomalous,
    enabling threshold-independent ranking comparisons.
    """

    name: str

    def fit_predict(self, X: NDArray[np.float64]) -> NDArray[np.bool_]: ...

    def score_samples(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...


def as_2d(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``X`` as a 2-D ``(n_samples, n_features)`` float array."""

    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr
