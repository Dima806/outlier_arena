"""The arena: detectors x anomaly types x dimensions x contamination.

This module wires the detectors, the planted-truth datasets and the metrics
together into the comparisons the notebooks render. It also ships the two
Notebook 06 conveniences: :func:`detect_outliers` (one detector) and the
agreement machinery re-exported from :mod:`src.evaluation.agreement`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.datasets.synthetic import ANOMALY_TYPES, make_dataset
from src.detectors import CANONICAL_SIX, make_detector
from src.evaluation.detection import detection_scores, recall_by_type


def detect_outliers(
    X: NDArray[np.float64], method: str = "isolation_forest", contamination: float = 0.08
) -> NDArray[np.bool_]:
    """Flag outliers in ``X`` with a single named detector (Notebook 06 utility)."""

    return make_detector(method, contamination=contamination).fit_predict(X)


def run_labels(
    X: NDArray[np.float64],
    methods: tuple[str, ...] | None = None,
    contamination: float = 0.08,
) -> dict[str, NDArray[np.bool_]]:
    """Run several detectors on ``X``; return ``method -> boolean outlier mask``."""

    methods = methods or CANONICAL_SIX
    return {m: make_detector(m, contamination=contamination).fit_predict(X) for m in methods}


@dataclass
class ArenaResult:
    """Per-detector scores on one dataset, plus the agreement report."""

    scores: pd.DataFrame  # one row per detector
    agreement: object  # AgreementReport (avoids an import cycle in the annotation)
    labels: dict[str, NDArray[np.bool_]]


def run_arena(
    n_samples: int = 1000,
    n_features: int = 2,
    contamination: float = 0.08,
    methods: tuple[str, ...] | None = None,
    seed: int = 7,
) -> ArenaResult:
    """Score every detector on one planted-anomaly dataset.

    Returns detection precision/recall/F1, per-type recall, and the pairwise
    agreement report -- the two halves of Notebook 04.
    """

    from src.evaluation.agreement import build_report

    methods = methods or CANONICAL_SIX
    data = make_dataset(
        n_samples=n_samples,
        n_features=n_features,
        contamination=contamination,
        seed=seed,
    )
    labels = run_labels(data.X, methods=methods, contamination=contamination)

    rows: list[dict[str, float | str | int]] = []
    for method, pred in labels.items():
        score = detection_scores(data.y, pred)
        row: dict[str, float | str | int] = {
            "detector": method,
            "precision": score.precision,
            "recall": score.recall,
            "f1": score.f1,
            "n_flagged": score.n_flagged,
        }
        for atype in ANOMALY_TYPES:
            row[f"recall_{atype}"] = recall_by_type(data.types, pred, atype)
        rows.append(row)

    return ArenaResult(
        scores=pd.DataFrame(rows).set_index("detector"),
        agreement=build_report(labels),
        labels=labels,
    )


def run_dimension_sweep(
    dimensions: list[int],
    methods: tuple[str, ...] | None = None,
    n_samples: int = 1000,
    contamination: float = 0.08,
    seed: int = 7,
) -> pd.DataFrame:
    """Recall vs feature dimension for each detector (Notebook 05).

    Distance-based detectors degrade as dimension rises (all points become
    nearly equidistant); Isolation Forest holds up better.
    """

    methods = methods or CANONICAL_SIX
    rows: list[dict[str, float | str | int]] = []
    for dim in dimensions:
        data = make_dataset(
            n_samples=n_samples, n_features=dim, contamination=contamination, seed=seed
        )
        labels = run_labels(data.X, methods=methods, contamination=contamination)
        for method, pred in labels.items():
            rows.append(
                {
                    "dimension": dim,
                    "detector": method,
                    "recall": detection_scores(data.y, pred).recall,
                }
            )
    return pd.DataFrame(rows)


def run_contamination_sweep(
    contaminations: list[float],
    methods: tuple[str, ...] | None = None,
    n_samples: int = 1000,
    n_features: int = 2,
    seed: int = 7,
) -> pd.DataFrame:
    """Precision/recall vs contamination rate for each detector (Notebook 05).

    As contamination rises the "anomalies are rare" assumption breaks and
    detectors that estimate a normal region get corrupted by the anomalies.
    """

    methods = methods or CANONICAL_SIX
    rows: list[dict[str, float | str | int]] = []
    for rate in contaminations:
        data = make_dataset(
            n_samples=n_samples, n_features=n_features, contamination=rate, seed=seed
        )
        labels = run_labels(data.X, methods=methods, contamination=rate)
        for method, pred in labels.items():
            score = detection_scores(data.y, pred)
            rows.append(
                {
                    "contamination": rate,
                    "detector": method,
                    "precision": score.precision,
                    "recall": score.recall,
                }
            )
    return pd.DataFrame(rows)
