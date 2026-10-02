"""Demonstrations of the two ways the three-sigma rule fails.

**Masking.** The mean and standard deviation are both inflated by extreme
values, so a cluster of extreme outliers pushes the ``mean + k*std`` threshold
*out past itself*. The rule then fails to flag the very points that broke it.
For ``m`` identical outliers among ``n`` points the outlier's z-score is
``sqrt((n - m) / m)`` regardless of how extreme the value is, so once
``m >= n / (k**2 + 1)`` the outliers can never be flagged -- masking is a
mathematical certainty, not an accident of the data.

**Swamping.** On right-skewed data the symmetric ``mean +/- k*std`` band
extends into the long tail and flags a crowd of legitimate points while
missing a genuine anomaly on the short side.

Both are shown against the robust IQR / modified z-score rules, which get them
right on the same data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.datasets.synthetic import make_skewed_1d
from src.detectors.statistical import IQRDetector, ThreeSigmaDetector


def make_masking_example(
    n_clean: int = 40,
    n_outliers: int = 8,
    outlier_value: float = 40.0,
    seed: int = 7,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """A 1-D sample whose extreme outliers mask themselves under three-sigma.

    Defaults put ``m = 8`` outliers among ``n = 48`` points, so each outlier's
    z-score is ``sqrt((48 - 8) / 8) ~= 2.24 < 3`` and three-sigma cannot flag
    them no matter how large ``outlier_value`` is. Returns ``(x, y)`` with
    ``y`` marking the planted outliers.
    """

    rng = np.random.default_rng(seed)
    clean = rng.normal(0.0, 1.0, n_clean)
    outliers = np.full(n_outliers, outlier_value)
    x = np.concatenate([clean, outliers])
    y = np.concatenate([np.zeros(n_clean, bool), np.ones(n_outliers, bool)])
    return x, y


@dataclass
class MaskingDemo:
    """Result of a masking or swamping demonstration."""

    x: NDArray[np.float64]
    y_true: NDArray[np.bool_]
    three_sigma_flags: NDArray[np.bool_]
    robust_flags: NDArray[np.bool_]
    three_sigma_lower: float
    three_sigma_upper: float
    # Recall on the planted anomalies, and false positives on clean points.
    three_sigma_recall: float
    robust_recall: float
    three_sigma_false_positives: int
    robust_false_positives: int


def _band(x: NDArray[np.float64], k: float) -> tuple[float, float]:
    mean, std = float(x.mean()), float(x.std(ddof=0))
    return mean - k * std, mean + k * std


def _summarise(
    x: NDArray[np.float64],
    y_true: NDArray[np.bool_],
    three_sigma: ThreeSigmaDetector,
    robust_flags: NDArray[np.bool_],
) -> MaskingDemo:
    ts_flags = three_sigma.fit_predict(x)
    lower, upper = _band(x, three_sigma.k)

    def recall(flags: NDArray[np.bool_]) -> float:
        n = int(np.sum(y_true))
        return float(np.sum(flags & y_true)) / n if n else 0.0

    return MaskingDemo(
        x=x,
        y_true=y_true,
        three_sigma_flags=ts_flags,
        robust_flags=robust_flags,
        three_sigma_lower=lower,
        three_sigma_upper=upper,
        three_sigma_recall=recall(ts_flags),
        robust_recall=recall(robust_flags),
        three_sigma_false_positives=int(np.sum(ts_flags & ~y_true)),
        robust_false_positives=int(np.sum(robust_flags & ~y_true)),
    )


def masking_demo(
    n_clean: int = 40,
    n_outliers: int = 8,
    outlier_value: float = 40.0,
    seed: int = 7,
) -> MaskingDemo:
    """Run the masking demonstration (three-sigma vs IQR)."""

    x, y = make_masking_example(
        n_clean=n_clean, n_outliers=n_outliers, outlier_value=outlier_value, seed=seed
    )
    return _summarise(x, y, ThreeSigmaDetector(), IQRDetector().fit_predict(x))


def swamping_demo(seed: int = 7) -> MaskingDemo:
    """Run the skew/swamping demonstration (three-sigma vs IQR).

    IQR's asymmetric fences hug the skewed data: its lower fence catches the
    short-side anomaly three-sigma misses. The modified z-score is available in
    :class:`ModifiedZScoreDetector` for comparison in the notebook.
    """

    x, y = make_skewed_1d(seed=seed)
    return _summarise(x, y, ThreeSigmaDetector(), IQRDetector().fit_predict(x))
