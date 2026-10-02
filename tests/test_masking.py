"""The three-sigma rule fails on the exact points it should catch.

These are the project's keystone assertions: the masking failure is a
mathematical certainty (not a quirk of one random seed), and the robust IQR
rule gets right what three-sigma gets wrong on the very same data.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.masking import make_masking_example, masking_demo, swamping_demo


def test_three_sigma_masks_the_outliers_that_break_it() -> None:
    demo = masking_demo()
    # Three-sigma flags NONE of the extreme planted outliers: its own inflated
    # std has moved the threshold out past them.
    assert demo.three_sigma_recall == 0.0
    # The robust IQR rule flags all of them on the same data.
    assert demo.robust_recall == 1.0


def test_masking_holds_no_matter_how_extreme_the_outliers() -> None:
    # The outlier value is irrelevant once there are enough of them: the
    # z-score of each outlier is sqrt((n - m) / m), independent of the value.
    for value in (30.0, 300.0, 3000.0, 30000.0):
        demo = masking_demo(outlier_value=value)
        assert demo.three_sigma_recall == 0.0


def test_masking_boundary_follows_the_z_score_formula() -> None:
    # With m outliers among n points, each outlier's z-score is
    # sqrt((n - m) / m). For m = 8, n = 48 that is ~2.236 < 3 -> masked.
    x, y = make_masking_example(n_clean=40, n_outliers=8, outlier_value=1000.0)
    z = np.abs((x - x.mean()) / x.std(ddof=0))
    expected = np.sqrt((48 - 8) / 8)
    # Approximate (not exact) because the clean points carry N(0, 1) noise
    # rather than sitting exactly at zero, but the formula governs the result.
    assert np.isclose(z[y].max(), expected, rtol=1e-2)
    assert expected < 3.0  # < 3 => the outliers can never clear the threshold


def test_three_sigma_swamps_the_tail_and_misses_the_short_side() -> None:
    demo = swamping_demo()
    # On right-skewed data three-sigma misses the genuine short-side anomaly...
    assert demo.three_sigma_recall == 0.0
    # ...while flagging legitimate long-tail points as false alarms (swamping).
    assert demo.three_sigma_false_positives > 0
    # The robust IQR rule catches the short-side anomaly three-sigma missed.
    assert demo.robust_recall == 1.0
