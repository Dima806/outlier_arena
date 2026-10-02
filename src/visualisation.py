"""Plot helpers shared by the notebooks and the Streamlit app.

Kept deliberately small: each function takes already-computed arrays/results
and returns a Matplotlib ``Figure`` so the notebooks stay orchestration-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # import matplotlib lazily so importing src stays cheap
    from matplotlib.figure import Figure

# A consistent colour for flagged (outlier) points across every figure.
_FLAG_COLOR = "#d1495b"
_NORMAL_COLOR = "#30638e"


def scatter_flags(
    X: NDArray[np.float64],
    flags: NDArray[np.bool_],
    title: str = "",
) -> Figure:
    """Scatter the first two features, highlighting flagged points."""

    import matplotlib.pyplot as plt

    arr = np.asarray(X, dtype=float)
    flags = np.asarray(flags, dtype=bool)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(arr[~flags, 0], arr[~flags, 1], s=12, c=_NORMAL_COLOR, alpha=0.5, label="normal")
    ax.scatter(arr[flags, 0], arr[flags, 1], s=36, c=_FLAG_COLOR, edgecolor="k", label="flagged")
    ax.set(title=title, xlabel="x0", ylabel="x1")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    return fig


def agreement_heatmap(names: list[str], matrix: NDArray[np.float64]) -> Figure:
    """Render the pairwise Jaccard agreement matrix as a heatmap."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_title("Pairwise detector agreement (Jaccard)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def masking_figure(
    x: NDArray[np.float64],
    y_true: NDArray[np.bool_],
    lower: float,
    upper: float,
    title: str = "three-sigma masking",
) -> Figure:
    """Show a 1-D sample with the three-sigma band and the planted anomalies."""

    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float)
    y_true = np.asarray(y_true, dtype=bool)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.scatter(x[~y_true], np.zeros(np.sum(~y_true)), s=14, c=_NORMAL_COLOR, alpha=0.5)
    ax.scatter(
        x[y_true],
        np.zeros(int(np.sum(y_true))),
        s=48,
        c=_FLAG_COLOR,
        edgecolor="k",
        zorder=3,
        label="planted anomaly",
    )
    ax.axvspan(lower, upper, color="0.85", label="inside 3σ band")
    ax.axvline(lower, color="0.4", ls="--", lw=1)
    ax.axvline(upper, color="0.4", ls="--", lw=1)
    ax.set(title=title, yticks=[], xlabel="value")
    ax.legend(loc="upper center", frameon=False, ncol=2)
    fig.tight_layout()
    return fig
