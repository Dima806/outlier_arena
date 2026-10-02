"""Detection quality against planted truth.

Accuracy is meaningless for rare anomalies -- a detector that flags nothing
scores ~99% when contamination is 1%. So this module reports precision,
recall and F1 (and recall broken down by anomaly type), following the metrics
project's lens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class DetectionScore:
    """Precision/recall/F1 of a detector against the planted-truth labels."""

    precision: float
    recall: float
    f1: float
    n_true: int
    n_flagged: int


def detection_scores(y_true: NDArray[np.bool_], y_pred: NDArray[np.bool_]) -> DetectionScore:
    """Precision, recall and F1 treating ``True`` as the anomaly (positive) class."""

    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return DetectionScore(
        precision=precision,
        recall=recall,
        f1=f1,
        n_true=int(np.sum(y_true)),
        n_flagged=int(np.sum(y_pred)),
    )


def recall_by_type(
    types: NDArray[np.str_],
    y_pred: NDArray[np.bool_],
    anomaly_type: str,
) -> float:
    """Recall restricted to points planted as ``anomaly_type``.

    This is how the project shows that different detectors catch different
    *kinds* of anomaly (LOF on local, Mahalanobis on multivariate, and so on).
    """

    mask = np.asarray(types) == anomaly_type
    n = int(np.sum(mask))
    if n == 0:
        return 0.0
    return float(np.sum(np.asarray(y_pred, dtype=bool)[mask])) / n
