"""Synthetic datasets with planted, known-type anomalies."""

from __future__ import annotations

from src.datasets.synthetic import (
    ANOMALY_TYPES,
    Dataset,
    make_dataset,
    make_skewed_1d,
)

__all__ = ["ANOMALY_TYPES", "Dataset", "make_dataset", "make_skewed_1d"]
