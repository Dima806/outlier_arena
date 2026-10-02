"""The six detectors disagree on most of the points they flag.

The headline figure (~20% consensus among flagged points) is configuration
dependent, so the assertions here use comfortable bands around the measured
values rather than pinning an exact number -- but they still encode the thesis:
the detectors substantially disagree, and the disputed set is real.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.datasets.synthetic import make_dataset
from src.evaluation.agreement import (
    build_report,
    consensus_fraction,
    mean_pairwise_agreement,
    pairwise_jaccard,
)
from src.evaluation.comparison import run_arena, run_labels


def _labels(seed: int = 7):
    data = make_dataset(n_samples=1000, contamination=0.08, seed=seed)
    return run_labels(data.X, contamination=0.08)


def test_consensus_is_about_one_fifth() -> None:
    # Across seeds the consensus among flagged points sits near 20%.
    fractions = [consensus_fraction(_labels(seed)) for seed in (1, 2, 7, 11, 42)]
    assert all(0.10 <= f <= 0.40 for f in fractions)


def test_detectors_substantially_disagree() -> None:
    labels = _labels()
    # Mean pairwise overlap is well below perfect agreement...
    assert 0.30 <= mean_pairwise_agreement(labels) <= 0.70
    # ...and at least one detector pair overlaps only weakly. (Note the two
    # univariate rules, three-sigma and IQR, instead coincide exactly here --
    # they both see only the global anomalies -- so some pairs do hit 1.0.)
    names, matrix = pairwise_jaccard(labels)
    off_diagonal = matrix[~np.eye(len(names), dtype=bool)]
    assert off_diagonal.min() < 0.5


def test_report_has_disputed_points() -> None:
    report = run_arena(n_samples=1000, contamination=0.08, seed=7).agreement
    assert report.n_disputed > 0
    # Disputed points are flagged by some detector but not all of them.
    assert report.disputed.sum() == report.n_disputed


def test_jaccard_matrix_is_symmetric_with_unit_diagonal() -> None:
    _, matrix = pairwise_jaccard(_labels())
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)


def test_agreement_vacuously_perfect_when_nothing_flagged() -> None:
    n = 50
    nothing = {m: np.zeros(n, dtype=bool) for m in ("a", "b", "c")}
    assert consensus_fraction(nothing) == pytest.approx(1.0)
    assert mean_pairwise_agreement(nothing) == pytest.approx(1.0)


def test_build_report_lists_all_detectors() -> None:
    labels = _labels()
    report = build_report(labels)
    assert set(report.names) == set(labels)
