"""Persist notebook outputs: figures as PNG, text/numbers as JSON.

Every notebook writes its visuals to ``outputs/figures/`` and its textual and
numerical results to ``outputs/data/`` through these two helpers,
so we can pull every figure and every quoted
number straight from disk. Paths are resolved from this file's location, so
saving works no matter what the current working directory is.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DATA_DIR = OUTPUTS_DIR / "data"


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas / dataclass values to JSON types."""

    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="index"))
    if isinstance(obj, pd.Series):
        return _jsonable(obj.to_dict())
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Sequence):
        return [_jsonable(v) for v in obj]
    return str(obj)  # last resort: stringify anything exotic


def save_figure(fig: Figure, name: str, dpi: int = 150) -> Path:
    """Save a Matplotlib figure to ``outputs/figures/<name>.png``."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_json(obj: Any, name: str) -> Path:
    """Save a (numpy/pandas-friendly) object to ``outputs/data/<name>.json``."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.json"
    path.write_text(json.dumps(_jsonable(obj), indent=2))
    return path
