# SPIRAL2 LLRF Fault Diagnosis Pipeline

A machine learning pipeline for detecting, classifying, and explaining faults in the
Low-Level RF (LLRF) control system of SPIRAL2's superconducting RF (SRF) cavities at GANIL,
from raw post-mortem acquisition buffers through to physics-grounded explainability.

Companion paper: see [`report/`](report/). Interactive pipeline walkthrough (data, features,
and modeling, with real figures): **[LLRF Pipeline Explorer](https://aghribi.github.io/LLRF_SP2_pipeline/)**.

## What it does

Raw LLRF post-mortem buffers are decoded (`programmes/PyPostMortem`), labeled by a
deterministic ground-truth fault classifier (7 fault categories + Normal), quality-filtered,
preprocessed (high-pass filter, per-file trigger alignment, Z-score normalization), and turned
into a several-hundred-feature matrix combining physics-informed and general statistical/
temporal descriptors. Five independent modeling tasks consume that matrix:

| Task | Approach | Result (verified, traced to `analysis/results_manifest/`) |
|---|---|---|
| Early fault-onset / precursor detection | Random Forest, fixed pre-trigger window only | 0.9991 ROC AUC (supervised); unsupervised baselines (Isolation Forest, LOF, Mahalanobis) confirm the separability is real, not a leak. **Open question, actively being investigated**: whether this reflects a genuine near-trigger signal or a stable per-capture characteristic — see [`report/sections/06_results.tex`](report/sections/06_results.tex) and the pipeline explorer's step 03. |
| Binary classification (fault vs. normal) | Logistic Regression / Random Forest / XGBoost / SVM, full event window | 97.3–100.0% accuracy, 0.9997–1.0000 ROC AUC. Near-ceiling by construction (the full window includes the interlock-trip transient) — not itself evidence of a leak; see the precursor task above for the harder, pre-trigger-only version. |
| Multi-label fault classification | Binary Relevance / Classifier Chains / Label Powerset, compared against deep multi-head architectures | See [`report/sections/06_results.tex`](report/sections/06_results.tex) for current per-method metrics. |
| Root cause identification | Multi-class classification on the same fault labels | 92.7% ± 0.8% accuracy (mean ± std, 10 repeated splits). |
| Fault subtype clustering | K-Means / Agglomerative / GMM, per fault category | 5 of 7 categories have enough events (≥20) to cluster meaningfully; best k = 2 for every one analyzed. |

An explainability layer (SHAP + LIME) provides physics-grounded interpretation of every
result, and every number above traces to a versioned entry in `analysis/results_manifest/`
rather than being hand-typed — see `report/code/check_no_hand_typed_numbers.py`.

**Current dataset**: V7, 2,865 events (1,478 Normal / 1,387 fault), 810 features — see
[`docs/`](docs/) for the full dataset lineage and what changed at each version. A dataset-wide
fix (a mislabeled header field, previously read as beam presence but actually meaning
feed-forward on/off) recovered ~756 legitimate events from 2019–2020 that an earlier,
incorrectly year-unscoped filter had discarded; propagating that fix through every downstream
model (beyond the binary/root-cause/precursor/clustering numbers above, which are the
last full re-run) is in progress.

## Repository structure

```
pipeline/           Production pipeline: extraction, preprocessing, feature engineering,
                     and the 10 modeling steps (00_scripts/), plus SLURM submission scripts
                     (01_slurm/) for cluster execution.
analysis/            Analysis notebooks/scripts and analysis/results_manifest/ — the
                     versioned source of truth every reported number traces to.
report/              LaTeX source for the companion paper: sections/, figures/ (real,
                     regenerable from pipeline output), code/ (figure + macro generation,
                     and the numeric-consistency checker).
docs/                Pipeline architecture, feature reference, and dataset-lineage docs.
utilities/           Supporting tools (data loading, diagnostics).
```

## Requirements

See `environment.yml` / `requirements_cluster.txt`. The pipeline was developed and run on a
Slurm-managed compute cluster with access to the raw LLRF post-mortem archive; it is not
runnable end-to-end without that raw data. The processed, derived feature dataset and every
result manifest are available separately (structured data release — see the paper for the
citation/DOI once published) for anyone wanting to reproduce the modeling steps without
access to the raw archive.

## Adapting this pipeline to another accelerator

This codebase currently has several SPIRAL2/GANIL/CC-IN2P3-specific assumptions (cluster
paths, the ground-truth classifier's own header field conventions) that a generalization pass
hasn't yet abstracted into configuration. See `docs/` and open issues for the current state of
that effort. Contributions toward making this more directly reusable by other facilities are
welcome — see `CONTRIBUTING.md`.

## License

See `LICENSE`.
