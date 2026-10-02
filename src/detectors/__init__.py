"""The six detectors, plus the modified z-score robust companion.

Every detector exposes the same small interface so the arena can treat them
uniformly:

* ``fit_predict(X) -> np.ndarray[bool]`` -- ``True`` marks an outlier.
* ``score_samples(X) -> np.ndarray[float]`` -- higher means *more* anomalous,
  so detectors can be compared by ranking, independent of any threshold.

The canonical six used for the agreement/arena results are
``three_sigma, iqr, mahalanobis, isolation_forest, lof, one_class_svm``.
``modified_zscore`` is the robust univariate companion the PRD includes
alongside IQR.
"""

from __future__ import annotations

from collections.abc import Callable

from src.detectors.base import Detector
from src.detectors.isolation_forest import IsolationForestDetector
from src.detectors.lof import LOFDetector
from src.detectors.mahalanobis import MahalanobisDetector
from src.detectors.one_class_svm import OneClassSVMDetector
from src.detectors.statistical import IQRDetector, ModifiedZScoreDetector, ThreeSigmaDetector

# The headline six (the agreement/arena set).
CANONICAL_SIX: tuple[str, ...] = (
    "three_sigma",
    "iqr",
    "mahalanobis",
    "isolation_forest",
    "lof",
    "one_class_svm",
)


def make_detector(method: str, contamination: float = 0.08) -> Detector:
    """Construct a detector by name with default hyper-parameters.

    ``contamination`` is passed to every detector that can use it; the
    univariate rules (which do not take a contamination input) ignore it.
    """

    factories: dict[str, Callable[[], Detector]] = {
        "three_sigma": ThreeSigmaDetector,
        "iqr": IQRDetector,
        "modified_zscore": ModifiedZScoreDetector,
        "mahalanobis": lambda: MahalanobisDetector(contamination=contamination),
        "isolation_forest": lambda: IsolationForestDetector(contamination=contamination),
        "lof": lambda: LOFDetector(contamination=contamination),
        "one_class_svm": lambda: OneClassSVMDetector(contamination=contamination),
    }
    if method not in factories:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(factories)}")
    return factories[method]()


__all__ = [
    "CANONICAL_SIX",
    "Detector",
    "IQRDetector",
    "IsolationForestDetector",
    "LOFDetector",
    "MahalanobisDetector",
    "ModifiedZScoreDetector",
    "OneClassSVMDetector",
    "ThreeSigmaDetector",
    "make_detector",
]
