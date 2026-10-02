"""From-scratch detectors match their reference implementations.

The project's credibility rests on the hand-built detectors being correct, so
here the from-scratch Mahalanobis is checked against sklearn's
``EmpiricalCovariance`` (they should be numerically identical) and the
from-scratch Isolation Forest intuition is checked for strong rank agreement
with sklearn's ``IsolationForest``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.covariance import EmpiricalCovariance

from src.datasets.synthetic import make_dataset
from src.detectors import CANONICAL_SIX, make_detector
from src.detectors.isolation_forest import isolation_depth_scores
from src.detectors.mahalanobis import MahalanobisDetector
from src.detectors.statistical import IQRDetector, ModifiedZScoreDetector, ThreeSigmaDetector


def test_mahalanobis_matches_empirical_covariance() -> None:
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([1.0, -2.0], [[3.0, 2.4], [2.4, 3.0]], size=400)
    mine = MahalanobisDetector().mahalanobis_sq(X)
    reference = EmpiricalCovariance().fit(X).mahalanobis(X)
    assert np.allclose(mine, reference, rtol=1e-6, atol=1e-6)


def test_isolation_forest_scratch_ranks_like_sklearn() -> None:
    data = make_dataset(n_samples=300, contamination=0.1, seed=3)
    scratch = isolation_depth_scores(data.X, n_trees=80, subsample=128, seed=1)
    sklearn_scores = make_detector("isolation_forest", contamination=0.1).score_samples(data.X)
    rho = spearmanr(scratch, sklearn_scores).correlation
    assert rho > 0.75


def test_three_sigma_flags_an_obvious_outlier() -> None:
    x = np.concatenate([np.zeros(20), np.ones(20), [50.0]])
    flags = ThreeSigmaDetector().fit_predict(x)
    assert flags[-1]
    assert flags.sum() == 1


def test_iqr_robust_to_a_single_extreme_value() -> None:
    # One extreme value must not blow up the quartiles the way it blows up std.
    x = np.concatenate([np.arange(50.0), [500.0]])
    flags = IQRDetector().fit_predict(x)
    assert flags[-1]


def test_modified_zscore_handles_zero_mad() -> None:
    # More than half the values identical -> MAD == 0; must not divide by zero.
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0])
    scores = ModifiedZScoreDetector().score_samples(x)
    assert np.all(np.isfinite(scores))
    assert scores[-1] == scores.max()


@pytest.mark.parametrize("method", CANONICAL_SIX)
def test_detector_interface(method: str) -> None:
    data = make_dataset(n_samples=200, contamination=0.1, seed=5)
    detector = make_detector(method, contamination=0.1)
    flags = detector.fit_predict(data.X)
    scores = detector.score_samples(data.X)
    assert flags.dtype == bool
    assert flags.shape == (data.X.shape[0],)
    assert scores.shape == (data.X.shape[0],)
    assert flags.any()  # something is flagged on contaminated data


def test_make_detector_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        make_detector("not_a_detector")
