"""outlier_arena -- an interactive tour of why six detectors disagree.

Four panels (per the PRD):

1. Masking demonstrator -- extreme points pull the three-sigma band out past
   themselves while IQR still flags them.
2. Anomaly-type sandbox -- see which detectors catch global/local/multivariate.
3. Agreement viewer -- the pairwise agreement matrix and the disputed points.
4. Dimension & contamination -- watch distance detectors degrade.

Run with ``make run`` (``streamlit run app/streamlit_app.py``) from the repo
root.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The library is a bare `src/` dir with no install step, so make the repo root
# importable regardless of how Streamlit launches this script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from src.datasets.synthetic import make_dataset  # noqa: E402
from src.detectors import CANONICAL_SIX  # noqa: E402
from src.evaluation.agreement import build_report  # noqa: E402
from src.evaluation.comparison import run_labels  # noqa: E402
from src.evaluation.detection import detection_scores, recall_by_type  # noqa: E402
from src.evaluation.masking import make_masking_example, masking_demo  # noqa: E402
from src.visualisation import agreement_heatmap, masking_figure, scatter_flags  # noqa: E402

st.set_page_config(page_title="outlier_arena", layout="wide")
st.title("outlier_arena")
st.caption("Six outlier detectors walk into your dataset. They disagree on ~80% of it.")

panel = st.sidebar.radio(
    "Panel",
    ["Masking", "Anomaly-type sandbox", "Agreement viewer", "Dimension & contamination"],
)


def _masking_panel() -> None:
    st.header("Three-sigma masks the outliers that break it")
    n_outliers = st.slider("Number of extreme outliers", 1, 20, 8)
    value = st.slider("How extreme (value)", 10.0, 200.0, 40.0, step=5.0)
    demo = masking_demo(n_outliers=n_outliers, outlier_value=value)
    x, y = make_masking_example(n_outliers=n_outliers, outlier_value=value)
    st.pyplot(masking_figure(x, y, demo.three_sigma_lower, demo.three_sigma_upper))
    c1, c2 = st.columns(2)
    c1.metric("three-sigma recall", f"{demo.three_sigma_recall:.0%}")
    c2.metric("IQR recall", f"{demo.robust_recall:.0%}")
    st.write(
        f"Three-sigma band: **[{demo.three_sigma_lower:.1f}, "
        f"{demo.three_sigma_upper:.1f}]** -- the outliers at {value:.0f} "
        "inflate the std until the band swallows them."
    )


def _sandbox_panel() -> None:
    st.header("Different detectors catch different kinds of anomaly")
    contamination = st.slider("Contamination", 0.02, 0.30, 0.08, step=0.01)
    data = make_dataset(n_samples=800, contamination=contamination, seed=7)
    labels = run_labels(data.X, contamination=contamination)
    method = st.selectbox("Detector", list(CANONICAL_SIX))
    st.pyplot(scatter_flags(data.X, labels[method], title=method))
    cols = st.columns(3)
    for col, atype in zip(cols, ("global", "local", "multivariate"), strict=True):
        col.metric(f"{atype} recall", f"{recall_by_type(data.types, labels[method], atype):.0%}")


def _agreement_panel() -> None:
    st.header("They agree on only about a fifth of the flagged points")
    contamination = st.slider("Contamination", 0.02, 0.30, 0.08, step=0.01)
    data = make_dataset(n_samples=800, contamination=contamination, seed=7)
    labels = run_labels(data.X, contamination=contamination)
    report = build_report(labels)
    st.pyplot(agreement_heatmap(report.names, report.jaccard_matrix))
    c1, c2 = st.columns(2)
    c1.metric("consensus among flagged", f"{report.consensus_fraction:.0%}")
    c2.metric("disputed points", report.n_disputed)
    st.pyplot(scatter_flags(data.X, report.disputed, title="points the detectors dispute"))


def _dimension_panel() -> None:
    st.header("Distance detectors degrade in high dimensions")
    dim = st.slider("Dimensions", 2, 100, 2)
    contamination = st.slider("Contamination", 0.02, 0.35, 0.08, step=0.01)
    data = make_dataset(n_samples=800, n_features=dim, contamination=contamination, seed=7)
    labels = run_labels(data.X, contamination=contamination)
    rows = {m: detection_scores(data.y, labels[m]).recall for m in CANONICAL_SIX}
    st.bar_chart(rows)
    st.caption(
        "Raise the dimension and watch Mahalanobis / LOF / one-class SVM fall "
        "while Isolation Forest holds up."
    )


_PANELS = {
    "Masking": _masking_panel,
    "Anomaly-type sandbox": _sandbox_panel,
    "Agreement viewer": _agreement_panel,
    "Dimension & contamination": _dimension_panel,
}
_PANELS[panel]()
