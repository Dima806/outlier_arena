"""Evaluation: detection quality, detector agreement, masking, and the arena."""

from __future__ import annotations

from src.evaluation.agreement import (
    AgreementReport,
    agreement_report,
    disputed_points,
    mean_pairwise_agreement,
    pairwise_jaccard,
)
from src.evaluation.comparison import ArenaResult, detect_outliers, run_arena, run_labels
from src.evaluation.detection import DetectionScore, detection_scores, recall_by_type
from src.evaluation.masking import (
    MaskingDemo,
    make_masking_example,
    masking_demo,
    swamping_demo,
)

__all__ = [
    "AgreementReport",
    "ArenaResult",
    "DetectionScore",
    "MaskingDemo",
    "agreement_report",
    "detect_outliers",
    "detection_scores",
    "disputed_points",
    "make_masking_example",
    "masking_demo",
    "mean_pairwise_agreement",
    "pairwise_jaccard",
    "recall_by_type",
    "run_arena",
    "run_labels",
    "swamping_demo",
]
