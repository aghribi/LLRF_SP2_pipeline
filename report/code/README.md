# Report Code (rewritten 2026-09-06)

This directory holds two things: the figure-generation scripts, and the
report-sync tooling (`generate_macros.py` / `check_no_hand_typed_numbers.py`,
documented in `../README.md`'s "Keeping results in sync with the pipeline"
section). This file covers figures only.

## Quick Start

From `report/`:

```bash
make figures
```

Or run one script directly (all figure scripts require `env_llrfspiral2`,
not the ambient `python3` -- they unpickle trained models built with
sklearn 1.6.1, and `fig_signal_examples.py` also imports `PyPostMortem`):

```bash
/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2/bin/python code/fig_precursor.py
```

## Scripts

| Script | Figure(s) | Real data source |
|---|---|---|
| `plot_style.py` | (shared style module, not a figure itself) | -- |
| `fig_signal_examples.py` | `signal_examples.pdf` | raw postmortem files via `PyPostMortem`, run through `prepare_data_cluster_v6.py`'s own `preprocess_signals()` |
| `fig_precursor.py` | `precursor_unsupervised_scores.pdf`, `precursor_roc_curves.pdf`, `precursor_feature_importance.pdf` | `cooked_data_v6/step_05_v6/precursor_detection.pkl` |
| `fig_subtype_profiles.py` | `subtype_feature_profiles.pdf` | `cooked_data_v6/step_09c_enhanced_subclass/enhanced_subclass_full.pkl` + `features_engineered.pkl` |
| `fig_shap_lime.py` | `shap_summary.pdf`, `lime_example.pdf` | retrains `prepare_10_enhanced_shap_analysis.py`'s per-label (one-vs-rest) Random Forest for one representative category |
| `fig_confusion_matrix.py` | `root_cause_confusion_matrix.pdf` | `cooked_data_v6/step_08_phase3a_v6/root_cause.pkl` |
| `pdf_to_png.py` | (dev helper, not part of the report) | renders a figure PDF to PNG for visual review -- not referenced by any section |

Every script's own docstring states its data source precisely; read that
before trusting a figure's provenance, don't assume from the filename alone.

## Style (`plot_style.py`)

All figures share one style via `apply_style()`:
- **Palette:** Okabe-Ito 8-color categorical (colorblind-safe by design);
  every categorical series also gets a distinct marker/linestyle, never
  color alone. Sequential/diverging plots use `cividis`/`PuOr`.
- **Font:** `STIXGeneral` body text, `stix` mathtext -- the closest available
  match to Computer-Modern-style scientific typesetting in this environment
  (no Times/CMU/Latin-Modern font is installed; checked directly against
  `matplotlib.font_manager`). `text.usetex` stays `False` throughout: no
  local LaTeX compile is invoked by these scripts.
- **Widths:** `SINGLE_COL_WIDTH = 3.404`in / `FULL_WIDTH = 7.058`in, measured
  by compiling a minimal document against `main.tex`'s live
  `revtex4-2` class and options (`reprint, aps, pra`) -- not assumed.
- **Output:** vector PDF (`pdf.fonttype: 42`, real embedded glyphs), written
  directly to `../figures/`.

## Data provenance -- do not reuse `analysis/outputs/`

Every figure must come from the current, corrected V6 pipeline
(`cooked_data_v6/`), never from `analysis/outputs/*.png` or `*.pkl` -- those
predate the leakage/methodology fixes documented in
`../sections/03_system_description.tex` and `04_methodology.tex` (the same
vintage as the fabricated numbers already found and removed from the report
text). If a script under `pipeline/00_scripts/` that a figure depends on
gets fixed or re-run, regenerate that figure (and re-run `make sync` in
`report/` to catch any number that changed).

## Adding a New Figure

1. Write `code/fig_<name>.py`: import `plot_style`, call `apply_style()`,
   load real data (state the exact source file/pickle in the docstring),
   build the plot, `savefig_pdf(fig, FIG_DIR / "<name>.pdf")`.
2. Add it to the `figures` target in `../Makefile`.
3. Reference it from a section: `\includegraphics[width=\linewidth]{figures/<name>.pdf}`
   (or `\textwidth` inside a `figure*` for a full-width figure).
4. Visually verify before committing: render to PNG with `pdf_to_png.py`
   and read it back, don't assume from the code alone that it's legible.
