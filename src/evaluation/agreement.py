"""How much the detectors (dis)agree on which points are outliers.

The project's headline is that the detectors concur on only about a fifth of
the outlier calls. We quantify that with the pairwise Jaccard overlap of the
detectors' flagged sets: for two detectors, the fraction of the points either
one flags that *both* flag. Averaged over all pairs this lands around 20% on
mixed-anomaly data -- so on ~80% of the flagged points at least one detector
disagrees.

The functions here operate on a plain mapping of ``name -> boolean mask`` so
they stay independent of how the labels were produced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _stack(labels: dict[str, NDArray[np.bool_]]) -> tuple[list[str], NDArray[np.bool_]]:
    names = list(labels)
    matrix = np.vstack([np.asarray(labels[n], dtype=bool) for n in names])
    return names, matrix


def _jaccard(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    union = int(np.sum(a | b))
    if union == 0:
        return 1.0  # neither flags anything -> perfect (vacuous) agreement
    return int(np.sum(a & b)) / union


def pairwise_jaccard(
    labels: dict[str, NDArray[np.bool_]],
) -> tuple[list[str], NDArray[np.float64]]:
    """Return detector names and their pairwise Jaccard agreement matrix."""

    names, matrix = _stack(labels)
    n = len(names)
    out = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = _jaccard(matrix[i], matrix[j])
    return names, out


def mean_pairwise_agreement(labels: dict[str, NDArray[np.bool_]]) -> float:
    """Mean off-diagonal pairwise Jaccard -- the headline agreement number."""

    _, mat = pairwise_jaccard(labels)
    n = mat.shape[0]
    if n < 2:
        return 1.0
    iu = np.triu_indices(n, k=1)
    return float(mat[iu].mean())


def consensus_fraction(labels: dict[str, NDArray[np.bool_]]) -> float:
    """Of all points flagged by at least one detector, the fraction flagged by all."""

    _, matrix = _stack(labels)
    flagged_by_any = np.any(matrix, axis=0)
    denom = int(np.sum(flagged_by_any))
    if denom == 0:
        return 1.0
    flagged_by_all = np.all(matrix, axis=0)
    return int(np.sum(flagged_by_all)) / denom


def disputed_points(labels: dict[str, NDArray[np.bool_]]) -> NDArray[np.bool_]:
    """Mask of points flagged by at least one detector but not by all of them."""

    _, matrix = _stack(labels)
    any_flag = np.any(matrix, axis=0)
    all_flag = np.all(matrix, axis=0)
    return any_flag & ~all_flag


@dataclass
class AgreementReport:
    """Everything the agreement viewer and Notebook 04 need."""

    names: list[str]
    jaccard_matrix: NDArray[np.float64]
    mean_agreement: float
    consensus_fraction: float
    disputed: NDArray[np.bool_]
    n_disputed: int


def build_report(labels: dict[str, NDArray[np.bool_]]) -> AgreementReport:
    """Assemble an :class:`AgreementReport` from a name -> mask mapping."""

    names, mat = pairwise_jaccard(labels)
    disputed = disputed_points(labels)
    return AgreementReport(
        names=names,
        jaccard_matrix=mat,
        mean_agreement=mean_pairwise_agreement(labels),
        consensus_fraction=consensus_fraction(labels),
        disputed=disputed,
        n_disputed=int(np.sum(disputed)),
    )


def agreement_report(
    X: NDArray[np.float64],
    methods: tuple[str, ...] | None = None,
    contamination: float = 0.08,
) -> AgreementReport:
    """Run several detectors on ``X`` and report where they conflict.

    Shipped for Notebook 06 as the "don't trust one verdict" utility. Imports
    the detectors lazily to keep this module dependency-light.
    """

    from src.evaluation.comparison import run_labels

    labels = run_labels(X, methods=methods, contamination=contamination)
    return build_report(labels)
