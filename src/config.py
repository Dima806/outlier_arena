"""Typed configuration loaded from ``config/settings.yaml``.

Contamination rates, anomaly-type weights, detector hyper-parameters and seeds
live in YAML rather than scattered through notebooks, so every number in the
project is traceable to one place and every run is reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# config/settings.yaml, resolved relative to the repository root.
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


class DatasetConfig(BaseModel):
    """Defaults for the synthetic planted-anomaly generator."""

    n_samples: int = 1000
    n_features: int = 2
    contamination: float = Field(0.08, ge=0.0, lt=0.5)
    type_weights: dict[str, float] = Field(
        default_factory=lambda: {"global": 0.34, "local": 0.33, "multivariate": 0.33}
    )


class DetectorConfig(BaseModel):
    """Thresholds and hyper-parameters for the six detectors."""

    three_sigma_k: float = 3.0
    iqr_k: float = 1.5
    modified_z_threshold: float = 3.5
    isolation_forest_estimators: int = 200
    lof_neighbors: int = 20
    one_class_svm_nu: float = Field(0.08, gt=0.0, le=1.0)


class StressConfig(BaseModel):
    """Grids for the high-dimension / high-contamination stress tests."""

    dimensions: list[int] = Field(default_factory=lambda: [2, 5, 10, 20, 50, 100])
    contaminations: list[float] = Field(default_factory=lambda: [0.02, 0.05, 0.10, 0.20, 0.35])


class Settings(BaseSettings):
    """Top-level project settings.

    Values come from ``config/settings.yaml`` by default; environment variables
    prefixed ``OUTLIER_ARENA_`` override them (nested keys use ``__``).
    """

    model_config = SettingsConfigDict(
        env_prefix="OUTLIER_ARENA_", env_nested_delimiter="__", extra="ignore"
    )

    seed: int = 7
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    detectors: DetectorConfig = Field(default_factory=DetectorConfig)
    stress: StressConfig = Field(default_factory=StressConfig)


def load_settings(path: Path | None = None) -> Settings:
    """Load :class:`Settings` from ``config/settings.yaml`` (or ``path``).

    Falls back to the model defaults when the file is absent, so the library
    still works without a config file present.
    """

    config_path = path or _CONFIG_PATH
    if not config_path.exists():
        return Settings()
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    return Settings(**raw)
