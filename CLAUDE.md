# Project: outlier_arena

> Head-to-head comparison of six outlier-detection methods on datasets with
> **planted, known-type anomalies**. The point of the project is a measured
> claim, not a vibe: the six detectors agree on only ~20% of points, the
> three-sigma rule every analyst reaches for is broken by both masking and
> skew, and the right detector depends entirely on the *kind* of anomaly —
> which nobody decides before running one.

This file is the working contract for anyone (human or Claude) working on this
repo. It is the source of truth for structure, conventions, and the claims the
code must actually demonstrate. When in doubt, re-read `.llm/PRD_outlier_arena.md`.

**Status: built and green.** `src/` library, `tests/` (29 passing), all six
notebooks, the Streamlit app and `config/` are implemented. `make ci` passes
(lint + typecheck + tests); `make notebooks` executes all six in ~50s and
regenerates `outputs/`. Keep it that way — run `make ci` before every commit,
and `make notebooks` when a notebook or anything it imports changes.

---

## 1. Thesis (what every notebook and test must defend)

The repo exists to *prove*, with reproducible numbers against planted truth,
five things. Do not weaken these into hedges; the whole value is that they are
measured, not asserted.

1. **Three-sigma is broken by the outliers it exists to catch (masking).**
   The mean and standard deviation are both wrecked by extreme values. One
   large outlier inflates `std` enough that `mean ± 3·std` moves *past* the
   outlier itself, so the rule fails to flag the very point that broke it.
   This is asserted in the test suite on the exact masked point.
2. **Three-sigma is broken by skew (swamping + miss).** It assumes a symmetric
   bell. On right-skewed data it flags a crowd of legitimate long-tail points
   (swamping) while missing the real anomaly on the short side.
3. **IQR and modified z-score are the robust drop-in fix nobody adopts.**
   Built on the median/quartiles, immune to masking, far better under skew.
   They flag correctly exactly where three-sigma fails, on the same data.
4. **Anomaly type determines detector.** Global (far from everything), local
   (abnormal for its neighborhood), multivariate (normal on every single
   feature, abnormal in combination) are genuinely different. Univariate rules
   see only global; Mahalanobis catches the multivariate cloud-violation; LOF
   catches local; Isolation Forest catches a mix. No detector wins everywhere.
5. **Detectors agree on only ~20% of points, and the disagreement is
   invisible.** Because almost everyone runs exactly one detector, they never
   see that a different detector would have removed a different fifth of the
   data. Outlier removal is a modeling decision made blindly by habit.

The constructive close (Notebook 06): decide the anomaly type first, use IQR
over plain three-sigma, run several detectors and inspect disagreements, and
treat removal as the deliberate modeling decision it is. **Never let the
takeaway collapse into "never remove outliers."**

---

## 2. Stack & hard constraints

- **Python 3.11+.** Package manager is **`uv` only** — never `pip`, never
  `conda`, never `poetry`. Install/run through `uv` and the Makefile.
- **Runtime budget: 2-CPU / 8 GB RAM GitHub Codespace, no GPU.** Everything
  must fit. Datasets are thousands to tens of thousands of points.
- **Each notebook runs end-to-end in < 5 minutes.** The arena cell (six
  detectors × anomaly types × dimensions × contamination) is the heaviest at
  ~4 min and must be cached.
- Runtime deps: `scikit-learn>=1.5`, `numpy>=1.26`, `scipy>=1.14`,
  `pandas>=2.2`, `matplotlib>=3.9`, `plotly>=5.22`, `streamlit>=1.38`,
  `pydantic>=2.9`, `pydantic-settings>=2.5`, `pyyaml>=6.0`.
- Dev deps: `pytest>=8.3`, `ruff>=0.8`, `ty>=0.0.1a7`, `jupyter>=1.1`,
  `ipykernel>=6.29`, `nbconvert>=7.16`, `nbclient>=0.10`.
- Do not add dependencies beyond this list without a stated reason. Keep the
  surface small; this is a from-principles teaching repo.

---

## 3. Repository layout

```
outlier_arena/
├── .devcontainer/devcontainer.json
├── .github/workflows/ci.yml
├── config/settings.yaml              # Contamination rates, anomaly types, seeds
├── notebooks/
│   ├── 01_three_sigma_is_broken.ipynb      # showpiece: masking + skew
│   ├── 02_kinds_of_outliers.ipynb          # taxonomy made visual
│   ├── 03_the_detector_zoo.ipynb           # each detector on the same data
│   ├── 04_arena.ipynb                      # payoff: precision-recall + agreement
│   ├── 05_high_dimensions_and_contamination.ipynb
│   └── 06_decision_framework.ipynb         # flowchart + detect_outliers util
├── src/
│   ├── config.py                     # pydantic-settings, loads config/settings.yaml
│   ├── detectors/
│   │   ├── statistical.py            # three-sigma, IQR, modified z-score — FROM SCRATCH
│   │   ├── mahalanobis.py            # multivariate distance — FROM SCRATCH
│   │   ├── isolation_forest.py       # sklearn + from-scratch isolation intuition
│   │   ├── lof.py                    # Local Outlier Factor (sklearn)
│   │   └── one_class_svm.py          # one-class SVM (sklearn)
│   ├── datasets/synthetic.py         # planted global/local/multivariate, tunable
│   ├── evaluation/
│   │   ├── agreement.py              # pairwise + overall agreement
│   │   ├── detection.py              # precision-recall vs planted truth
│   │   ├── masking.py                # masking + swamping demonstration
│   │   └── comparison.py             # arena harness + detect_outliers / run_labels / sweeps
│   ├── visualisation.py
│   └── artifacts.py                  # save_figure (PNG) + save_json — notebook outputs
├── app/streamlit_app.py
├── tests/
│   ├── test_detectors.py             # from-scratch match sklearn/reference
│   ├── test_masking.py               # three-sigma misses the masked outlier (asserted)
│   ├── test_agreement.py
│   └── test_detection.py
├── outputs/figures/*.png             # regenerated by `make notebooks`
├── outputs/data/*.json               # regenerated by `make notebooks`
├── pyproject.toml / uv.lock / Makefile / README.md / LICENSE (Apache-2.0)
```

- `src/` is the library; notebooks and the app import from it and stay thin.
  Logic lives in `src/`, not in notebook cells. A notebook cell orchestrates
  and visualizes; it does not define a detector.
- British spelling `visualisation.py` for the module — match it exactly in
  imports.
- **Imports are `from src...`.** The library is a bare `src/` package with no
  install step (no `[build-system]` → `uv` treats it as a virtual project), so
  the repo root must be importable: pytest gets it via `pythonpath = ["."]`,
  the Streamlit app via a small `sys.path` bootstrap at its top, and notebooks
  run from the repo root. Python is pinned to 3.12 (`.python-version`) for the
  sklearn/scipy wheels; `pytest-cov` is in dev deps for `make test-cov`.
- **Every notebook saves its outputs**: visuals via `save_figure(fig, name)` →
  `outputs/figures/<name>.png`, and all textual/numerical results via
  `save_json(obj, name)` → `outputs/data/<name>.json` (both from
  `src/artifacts.py`, which handles numpy/pandas/dataclasses). Each notebook
  also writes a `0N_summary.json` with its claims and headline numbers. Name
  outputs with the `0N_` notebook prefix. `make notebooks` runs `--inplace`.

---

## 4. Commands (always via Make)

```
make setup       # first-time: install uv, uv sync --all-extras, register kernel
make sync        # uv sync --all-extras
make lint        # format + check + typecheck (run before every commit)
make format      # ruff format
make check       # ruff check --fix
make typecheck   # ty check src/
make test        # pytest
make test-cov    # pytest with coverage
make notebooks   # execute all notebooks headless (300s timeout each)
make run         # streamlit app on :8501
make lab         # jupyterlab on :8888
make ci          # sync lint test  (what CI runs)
make dev         # lint test  (fast local loop)
make clean / make reset
```

`make lint` and `make test` must pass with **zero errors** at every commit.
`make notebooks` must succeed end-to-end — a notebook that errors or exceeds
300s is a failure.

---

## 5. Detectors — build spec

Six detectors, each exposing a consistent interface (predict outlier labels
and, where meaningful, a continuous score so ranking can be evaluated
independently of any threshold).

| Detector | File | Finds | Notes |
|---|---|---|---|
| Three-sigma | `statistical.py` | global univariate | **from scratch**; the cautionary protagonist. |
| IQR (box-plot) | `statistical.py` | global univariate | **from scratch**; 1.5×IQR; robust drop-in. |
| Modified z-score | `statistical.py` | global univariate | **from scratch**; MAD-based robust standard. |
| Mahalanobis | `mahalanobis.py` | multivariate cloud-violation | **from scratch**; covariance-adjusted distance. |
| Isolation Forest | `isolation_forest.py` | mixed / general tabular | sklearn, preceded by a from-scratch isolation-depth intuition. |
| Local Outlier Factor | `lof.py` | local density | sklearn. |
| One-class SVM | `one_class_svm.py` | boundary | sklearn; included for its different failure profile. |

Rules:
- **From-scratch means numpy/scipy only** — no sklearn shortcut inside the
  from-scratch path. Each from-scratch detector is validated against its
  sklearn/reference counterpart within tolerance in `test_detectors.py`.
- Give each detector its intended contamination input **and** expose raw
  scores, so the arena can also rank-score independent of the threshold (this
  is how we compare fairly — see PRD risk table).
- Standardize features before distance-based detection (Mahalanobis, LOF,
  one-class SVM); call this out in code and notebook.

---

## 6. Datasets — planted truth is the whole methodology

`src/datasets/synthetic.py` generates **purely synthetic** data because every
claim depends on knowing which points are truly anomalous and of what type.

- Produces clean base data plus planted **global**, **local**, and
  **multivariate** anomalies, each tagged with its type and location.
- Tunable **contamination rate** and **dimension**.
- Includes **skewed and clustered** clean distributions so three-sigma's
  failure modes appear on realistic data.
- Everything is **seeded** (seeds live in `config/settings.yaml`). Same seed →
  same data → same numbers. Reproducibility is non-negotiable.

Because type and location are known, the project **measures** rather than
asserts: masking on the exact masked point, per-detector precision/recall vs
true labels, agreement rate, and how all of it shifts with dimension and
contamination.

---

## 7. Evaluation — the metrics lens

- **`detection.py`:** precision and recall against planted truth. **Report
  precision-recall, never accuracy** — anomalies are rare, so accuracy is
  meaningless (a detector flagging nothing scores ~99%). This follows the
  metrics project's lens.
- **`agreement.py`:** the ~20% headline maps to **`consensus_fraction`** — of
  the points flagged by *any* detector, the fraction flagged by *all* six
  (measured ≈ 0.22). `mean_pairwise_agreement` (mean pairwise Jaccard ≈ 0.48)
  is the secondary number; don't confuse the two. Note the two univariate
  rules (three-sigma, IQR) coincide exactly (Jaccard 1.0) — they see only the
  global anomalies, so `test_agreement` asserts a *low min* off-diagonal, not
  "all pairs < 1".
- **`masking.py`:** demonstrates masking (threshold moves past the outlier)
  and swamping (legit tail points flagged) quantitatively.
- **`comparison.py`:** the arena harness — detectors × anomaly types ×
  dimensions × contamination, cached because it is the heaviest computation.

---

## 8. Notebooks — narrative order

Each notebook is a self-contained argument that imports from `src/`, runs in
< 5 min, is seeded, and saves figures to `outputs/figures/`.

1. **01 — Three-Sigma Is Broken (showpiece).** Masking first (outlier sits
   inside its own inflated threshold), then skew (swamp the tail, miss the
   short side), then IQR + modified z-score doing both correctly on the same
   data. One figure should show the failure and the robust fix together.
2. **02 — Kinds of Outliers.** Global/local/multivariate drawn in 2D; the
   multivariate point that is normal on the x-histogram, normal on the
   y-histogram, and obviously wrong in the scatter. Name the anomaly before
   choosing the detector.
3. **03 — The Detector Zoo.** Each detector on the same all-types data with
   flagged points highlighted; from-scratch Isolation Forest intuition and
   from-scratch Mahalanobis validated against reference.
4. **04 — The Arena (payoff).** Precision-recall against planted truth, then
   the agreement matrix landing near 20%. Headline: your detector choice
   silently decides which fifth gets removed, and detectors mostly disagree.
5. **05 — High Dimensions & Contamination.** Distance *concentration* (the
   fundamental curse), then the honest result: dimensionality **helps when the
   extra features carry signal, hurts when they are noise** — and when it
   hurts, Isolation Forest degrades too (random splits land on noise features).
   High contamination corrupts rarity-assuming detectors. **Do not "fix" this
   back to the PRD's "distance degrades, IF holds up" line** — on this planted
   data that is not reproducible; the signal-vs-noise framing is what the
   numbers actually support. Warnings: standardize, cut noise dims, know your
   contamination rate.
6. **06 — Decision Framework.** The flowchart + meta-rules. Ships
   `detect_outliers(X, method, contamination)` and an `agreement_report` that
   runs several detectors and surfaces conflicts.

---

## 9. Coding conventions

- **Ruff** governs style: `line-length = 99`, `target-version = "py311"`,
  lint select `["E","F","W","I","UP","N","B","A","SIM","PTH"]`, ignore
  `["E501","N803","N806"]` (N803/N806 ignored so `X` — the standard feature-
  matrix name, used in the public `detect_outliers(X, ...)` API — is allowed).
  Run `make format` then `make check`. Use `pathlib` (PTH), not `os.path`.
- **`ty`** type-checks `src/`. Add type hints to all public functions in
  `src/`; keep them honest (no blanket `Any` to silence the checker).
- **`pydantic` / `pydantic-settings`** for config — `config/settings.yaml` is
  loaded through `src/config.py`. No magic numbers scattered in notebooks:
  contamination rates, anomaly specs, and seeds come from config.
- Keep functions small and named for what they demonstrate. This is teaching
  code — clarity beats cleverness. Comment the *why* (the statistical reason),
  not the *what*.
- Deterministic everywhere: thread a seed through generation and any detector
  with randomness (Isolation Forest). No unseeded `np.random` calls.

---

## 10. Testing — the claims are asserted, not hoped for

`tests/` encodes the thesis so it cannot silently regress:

- `test_masking.py` — three-sigma **misses** the masked outlier that inflated
  its own threshold (hard assertion), and IQR flags it.
- `test_detectors.py` — each from-scratch detector matches its
  sklearn/reference counterpart within tolerance.
- `test_detection.py` — LOF has higher recall than global rules on the
  local-anomaly type; Mahalanobis detects the per-feature-normal,
  combination-abnormal point.
- `test_agreement.py` — overall agreement on mixed-anomaly data lands in the
  expected (~20%) range for the stated configuration.

Tests use fixed seeds. If a test encodes a numeric threshold, state the exact
config it depends on (the ~20% figure is config-dependent; the masking failure
is exact).

---

## 11. Success criteria (from the PRD — the bar for "done")

- Overall agreement ≈ 20%, quantified. • Masking failure asserted on the exact
  point. • Skew swamp + short-side miss quantified. • IQR/modified-z correct
  where three-sigma fails. • Mahalanobis catches the multivariate point. • LOF
  out-recalls global rules on local anomalies. • From-scratch matches reference
  within tolerance. • Distances concentrate with dimension; detection degrades
  when the extra dimensions are noise (see Notebook 05 note — this replaces the
  PRD's "IF more stable" criterion, which the data does not support). • High
  contamination degrades rarity-assuming detectors. • Precision-recall (not
  accuracy) reported. • Every notebook < 5 min on 2-CPU. • `make lint` and
  `make test` pass clean.

---

## 12. Working style / token rules

- **Caveman mode: ON.** Terse, code-first replies. Lead with the change, not a
  preamble. Don't re-explain the thesis back; it's in this file.
- Prefer editing `src/` and letting notebooks/app import it over writing long
  cells.
- `/compact` at notebook boundaries.
- Before committing: `make lint && make test` green, and if notebooks changed,
  `make notebooks` green.

### Do
- Keep claims measured and reproducible; cite the exact config behind any number.
- Build the from-scratch detectors from numpy/scipy and validate against sklearn.
- Standardize before distance methods; report precision-recall for rare events.

### Don't
- Don't use `pip`; don't add deps beyond §2 without reason.
- Don't put detector logic in notebook cells.
- Don't let the message become "never remove outliers" — the framework is
  constructive.
- Don't introduce unseeded randomness or accuracy-based evaluation.
```
