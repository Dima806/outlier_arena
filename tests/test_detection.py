"""Each detector catches the kind of anomaly it is built for.

Against the planted-truth labels we check the claims from the detector zoo:
univariate rules see only global anomalies, LOF catches local ones the global
rules cannot, and Mahalanobis catches the per-feature-normal, combination-
abnormal point. Scores are precision/recall (never accuracy), the right lens
for rare events.
"""

from __future__ import annotations

import numpy as np

from src.datasets.synthetic import make_dataset
from src.detectors import make_detector
from src.evaluation.comparison import run_labels
from src.evaluation.detection import detection_scores, recall_by_type


def _predictions(contamination: float = 0.08, seed: int = 7):
    data = make_dataset(n_samples=1000, contamination=contamination, seed=seed)
    labels = run_labels(data.X, contamination=contamination)
    return data, labels


def test_detection_scores_on_a_known_example() -> None:
    y_true = np.array([True, True, True, False, False])
    y_pred = np.array([True, False, True, True, False])
    score = detection_scores(y_true, y_pred)
    assert score.precision == 2 / 3  # 2 TP, 1 FP
    assert score.recall == 2 / 3  # 2 TP, 1 FN
    assert score.n_true == 3
    assert score.n_flagged == 3


def test_accuracy_would_lie_but_recall_does_not() -> None:
    # A detector that flags nothing is ~92% "accurate" at 8% contamination,
    # yet its recall is zero -- which is why the project reports recall.
    data, _ = _predictions()
    flags_nothing = np.zeros(data.X.shape[0], dtype=bool)
    score = detection_scores(data.y, flags_nothing)
    assert score.recall == 0.0
    accuracy = np.mean(flags_nothing == data.y)
    assert accuracy > 0.9


def test_univariate_rules_see_only_global_anomalies() -> None:
    data, labels = _predictions()
    for rule in ("three_sigma", "iqr"):
        assert recall_by_type(data.types, labels[rule], "global") >= 0.9
        assert recall_by_type(data.types, labels[rule], "local") < 0.2
        assert recall_by_type(data.types, labels[rule], "multivariate") < 0.2


def test_lof_beats_global_rules_on_local_anomalies() -> None:
    data, labels = _predictions()
    lof_local = recall_by_type(data.types, labels["lof"], "local")
    three_sigma_local = recall_by_type(data.types, labels["three_sigma"], "local")
    assert lof_local >= 0.3
    assert lof_local > three_sigma_local


def test_mahalanobis_catches_the_multivariate_point() -> None:
    data, labels = _predictions()
    maha_mv = recall_by_type(data.types, labels["mahalanobis"], "multivariate")
    three_sigma_mv = recall_by_type(data.types, labels["three_sigma"], "multivariate")
    assert maha_mv >= 0.5
    assert maha_mv > three_sigma_mv


def test_every_detector_catches_global_anomalies() -> None:
    data, labels = _predictions()
    for method in labels:
        assert recall_by_type(data.types, labels[method], "global") >= 0.7


def test_detect_outliers_single_method_runs() -> None:
    data, _ = _predictions()
    flags = make_detector("isolation_forest", contamination=0.08).fit_predict(data.X)
    assert flags.dtype == bool
    assert flags.sum() > 0
