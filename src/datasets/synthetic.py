"""Purely synthetic data with planted anomalies of known type.

The whole argument of the project depends on knowing which points are truly
anomalous and *what kind* they are, so everything here is generated and
labelled rather than observed.

Clean data is two correlated Gaussian clusters, which gives three things at
once: a strong global correlation structure (so a multivariate outlier can
violate it), dense neighbourhoods (so a local outlier can sit in a sparse
pocket), and a wide global range (so a local/multivariate outlier can stay
inside every feature's marginal range and thus be invisible to univariate
rules).

Three anomaly types are planted:

* ``global``       -- far from everything in every feature.
* ``local``        -- lands in the sparse gap between the clusters; normal on
  each feature and near the global centre, abnormal only for its neighbourhood.
* ``multivariate`` -- normal on each feature's marginal range but off the
  correlation axis of the cloud.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Labels used for the planted-truth ``types`` array. "normal" marks clean points.
ANOMALY_TYPES: tuple[str, ...] = ("global", "local", "multivariate")

_DEFAULT_WEIGHTS = {"global": 0.34, "local": 0.33, "multivariate": 0.33}


@dataclass
class Dataset:
    """A generated dataset with planted-truth labels.

    Attributes
    ----------
    X:
        ``(n_samples, n_features)`` feature matrix.
    y:
        Boolean anomaly mask (``True`` == planted anomaly).
    types:
        Per-point type label: ``"normal"`` or one of :data:`ANOMALY_TYPES`.
    """

    X: NDArray[np.float64]
    y: NDArray[np.bool_]
    types: NDArray[np.str_]
    feature_names: list[str] = field(default_factory=list)

    def type_mask(self, anomaly_type: str) -> NDArray[np.bool_]:
        """Boolean mask of points planted as ``anomaly_type``."""

        return self.types == anomaly_type


def _correlation_matrix(n_features: int, rho: float) -> NDArray[np.float64]:
    """Compound-symmetry correlation matrix (``1`` on diagonal, ``rho`` off)."""

    r = np.full((n_features, n_features), rho)
    np.fill_diagonal(r, 1.0)
    return r


def _cluster(
    rng: np.random.Generator,
    n: int,
    center: NDArray[np.float64],
    scale: float,
    rho: float,
) -> NDArray[np.float64]:
    cov = (scale**2) * _correlation_matrix(len(center), rho)
    return rng.multivariate_normal(center, cov, size=n)


def make_dataset(
    n_samples: int = 1000,
    n_features: int = 2,
    contamination: float = 0.08,
    type_weights: dict[str, float] | None = None,
    seed: int = 7,
    rho: float = 0.85,
) -> Dataset:
    """Generate a dataset with planted global/local/multivariate anomalies.

    Parameters mirror ``config/settings.yaml``. ``contamination`` is the
    fraction of points that are anomalies; ``type_weights`` splits that budget
    across the three types.
    """

    if not 0.0 <= contamination < 0.5:
        raise ValueError("contamination must be in [0, 0.5)")
    weights = type_weights or _DEFAULT_WEIGHTS
    rng = np.random.default_rng(seed)

    n_anom = int(round(n_samples * contamination))
    n_clean = n_samples - n_anom

    # Two correlated clusters along the all-ones diagonal; B is offset so a gap
    # opens between them. Having two clusters also widens the global per-feature
    # range, which is what lets local/multivariate anomalies stay inside every
    # feature's marginal range (and so stay invisible to univariate rules).
    ones = np.ones(n_features) / np.sqrt(n_features)
    scale_a, scale_b = 1.6, 1.1
    center_a = np.zeros(n_features)
    center_b = ones * 16.0
    n_a = n_clean // 2
    n_b = n_clean - n_a
    clean = np.vstack(
        [
            _cluster(rng, n_a, center_a, scale=scale_a, rho=rho),
            _cluster(rng, n_b, center_b, scale=scale_b, rho=rho),
        ]
    )

    # Per-feature spread of the clean data, used to size global anomalies.
    feat_mean = clean.mean(axis=0)
    feat_std = clean.std(axis=0, ddof=0)
    feat_min, feat_max = clean.min(axis=0), clean.max(axis=0)

    # Split the contamination budget across types.
    counts = _split_counts(n_anom, weights)
    anomalies: list[NDArray[np.float64]] = []
    labels: list[str] = []

    # Global: far outside every feature's range (>= 5 std from the grand mean).
    for _ in range(counts["global"]):
        signs = rng.choice([-1.0, 1.0], size=n_features)
        anomalies.append(feat_mean + signs * feat_std * rng.uniform(5.0, 7.0, n_features))
        labels.append("global")

    # Local: scattered individually through the sparse region away from both
    # dense clusters. Because anomalies are rare, each one's neighbourhood is
    # dominated by dense clean points, so its relative density is low (LOF's
    # target) -- but it stays inside every feature's range, so univariate rules
    # never see it. Scattering (rather than clumping) is essential: a clump of
    # local anomalies would look like a dense cluster of its own.
    reject_radius = 3.5 * scale_a
    for _ in range(counts["local"]):
        anomalies.append(
            _sparse_point(rng, feat_min, feat_max, (center_a, center_b), reject_radius)
        )
        labels.append("local")

    # Multivariate: within each feature's marginal range but off the correlation
    # axis. Take a cluster-A point and push it perpendicular to the all-ones
    # diagonal by a few *cluster* widths -- enough to violate the correlation,
    # not so far that any single feature becomes extreme.
    for _ in range(counts["multivariate"]):
        base = _cluster(rng, 1, center_a, scale=scale_a, rho=rho)[0]
        perp = _random_perpendicular(rng, ones)
        anomalies.append(base + perp * scale_a * rng.uniform(3.0, 5.0))
        labels.append("multivariate")

    anom_arr = np.vstack(anomalies) if anomalies else np.empty((0, n_features))
    X = np.vstack([clean, anom_arr])
    types = np.array(["normal"] * n_clean + labels)
    y = types != "normal"

    # Shuffle so anomalies are not all at the end.
    order = rng.permutation(X.shape[0])
    return Dataset(
        X=X[order],
        y=y[order],
        types=types[order],
        feature_names=[f"x{i}" for i in range(n_features)],
    )


def _sparse_point(
    rng: np.random.Generator,
    low: NDArray[np.float64],
    high: NDArray[np.float64],
    centers: tuple[NDArray[np.float64], ...],
    reject_radius: float,
    max_tries: int = 200,
) -> NDArray[np.float64]:
    """Sample a point inside the data's bounding box but far from every cluster.

    Rejection sampling keeps candidates that are at least ``reject_radius`` from
    all cluster centres, placing them in the sparse region where a point is a
    local outlier yet still inside each feature's marginal range.
    """

    for _ in range(max_tries):
        candidate = rng.uniform(low, high)
        if all(np.linalg.norm(candidate - c) > reject_radius for c in centers):
            return candidate
    return candidate  # fall back to the last draw if the box is crowded


def _random_perpendicular(
    rng: np.random.Generator, axis: NDArray[np.float64]
) -> NDArray[np.float64]:
    """A unit vector perpendicular to ``axis`` (which must be unit length)."""

    v = rng.normal(size=axis.shape)
    v -= np.dot(v, axis) * axis
    norm = np.linalg.norm(v)
    if norm < 1e-12:  # degenerate draw; retry deterministically
        v = np.zeros_like(axis)
        v[0] = 1.0
        v -= np.dot(v, axis) * axis
        norm = np.linalg.norm(v)
    return v / norm


def _split_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split ``total`` anomalies across types by ``weights`` (largest-remainder)."""

    w_sum = sum(weights.get(t, 0.0) for t in ANOMALY_TYPES)
    if w_sum <= 0:
        w_sum = 1.0
    raw = {t: total * weights.get(t, 0.0) / w_sum for t in ANOMALY_TYPES}
    counts = {t: int(np.floor(v)) for t, v in raw.items()}
    remainder = total - sum(counts.values())
    # Hand out the leftover to the largest fractional parts.
    order = sorted(ANOMALY_TYPES, key=lambda t: raw[t] - counts[t], reverse=True)
    for t in order[:remainder]:
        counts[t] += 1
    return counts


def make_skewed_1d(
    n_samples: int = 500,
    seed: int = 7,
    short_side_value: float = -1.4,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Right-skewed 1-D data plus one planted short-side anomaly.

    Used by Notebook 01 to show three-sigma's two skew failures at once:

    * **Swamping** -- the long right tail inflates the standard deviation, yet
      the symmetric ``mean + 3*std`` upper bound still cuts into legitimate
      tail points and flags them as outliers.
    * **Short-side miss** -- the same inflated std pushes the *lower* bound far
      below the data (into values the log-normal never produces), so a genuine
      anomaly on the short side sits comfortably inside the band and is missed.

    The default ``short_side_value`` is chosen to land inside three-sigma's
    lower bound but outside the robust IQR lower fence, so IQR catches exactly
    what three-sigma misses. Returns ``(x, y)`` with ``y`` marking the anomaly.
    """

    rng = np.random.default_rng(seed)
    # Log-normal is right-skewed: a long right tail of legitimate points.
    x = rng.lognormal(mean=0.0, sigma=0.75, size=n_samples)
    x = np.append(x, short_side_value)  # the real anomaly, on the short side
    y = np.zeros(x.shape[0], dtype=bool)
    y[-1] = True
    order = rng.permutation(x.shape[0])
    return x[order], y[order]
