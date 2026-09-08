"""
Results-manifest helper for SPIRAL2 anomalies_exploration.

Why this exists: the 2026-09-03 audit (NOTEBOOK_AUDIT_2026-09-03.md) found that the
report's headline numbers largely traced to stale notebook markdown text that was
never reconciled with the notebooks' own re-executed code output. This helper closes
that gap: a notebook's final cell calls save_manifest(...) after computing its real
results, which writes a git-tracked, human-diffable YAML file AND displays the same
values as that cell's output -- guaranteed fresh because the notebook is executed
top-to-bottom (nbconvert --execute) each time it's finalized. Prose in the notebook
and in the report should cite these values (or, in the report, a LaTeX macro
generated from them -- see report/code/generate_macros.py), never a hand-typed number.

Usage, as the last cell of a canonical notebook:

    from utilities.reporting.manifest import save_manifest
    save_manifest(
        phase="06_binary_classification",
        metrics={
            "xgboost_accuracy": {"value": 1.0, "fmt": ".1%", "label": "XGBoost accuracy"},
            "xgboost_roc_auc":  {"value": 1.0, "fmt": ".4f", "label": "XGBoost ROC AUC"},
        },
        pipeline_run={
            "dataset_version": "V6",
            "dataset_path": "/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6/features_engineered_v6.pkl",
            "slurm_job": "57924849",
            "script": "pipeline/00_scripts/prepare_06_phase1_binary.py",
        },
    )
"""

from datetime import datetime, timezone
from pathlib import Path

import yaml

MANIFEST_DIR = Path(__file__).resolve().parents[2] / "analysis" / "results_manifest"


def _format_value(value, fmt):
    if fmt is None:
        return str(value)
    try:
        return format(value, fmt)
    except (ValueError, TypeError):
        return str(value)


def save_manifest(phase, metrics, pipeline_run, meta=None, stability=None):
    """
    Write analysis/results_manifest/<phase>.yaml and return the same dict, so the
    calling notebook cell can also `display()` it as its own output.

    phase        : short slug, e.g. "06_binary_classification" -> <phase>.yaml
    metrics      : dict of {metric_name: {"value": ..., "fmt": "<format spec or None>",
                   "label": "human-readable label"}}
    pipeline_run : dict identifying exactly which dataset/run produced these numbers
                   (dataset_version, dataset_path, slurm_job, script -- non-optional,
                   see module docstring for why).
    meta         : optional extra free-form metadata (e.g. notes, caveats).
    stability    : optional dict from leakage_safe_features.repeated_leakage_safe_eval(),
                   i.e. {metric_name: {"mean","std","min","max","values"} or
                   {"per_label": {label: {"mean","std","min","max","values"}}}}.
                   Written through as-is (already all built-in types) alongside a
                   top-level "n_repeats" inferred from the first metric's values,
                   so a report macro can cite "mean ± std over N repeated splits"
                   instead of a single fixed-seed number.
    """
    if not metrics:
        raise ValueError("metrics must be non-empty")
    required_run_keys = {"dataset_version", "dataset_path", "script"}
    missing = required_run_keys - set(pipeline_run)
    if missing:
        raise ValueError(f"pipeline_run missing required keys: {sorted(missing)}")

    for name, m in metrics.items():
        if "value" not in m or "label" not in m:
            raise ValueError(f"metric '{name}' must have at least 'value' and 'label'")
        m.setdefault("fmt", None)
        m["display"] = _format_value(m["value"], m["fmt"])

    manifest = {
        "phase": phase,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_run": pipeline_run,
        "metrics": metrics,
    }
    if meta:
        manifest["meta"] = meta
    if stability:
        n_repeats = None
        for m in stability.values():
            values = m.get("values")
            if values is not None:
                n_repeats = len(values)
                break
            per_label = m.get("per_label")
            if per_label:
                any_label = next(iter(per_label.values()))
                n_repeats = len(any_label["values"])
                break
        manifest["stability"] = {"n_repeats": n_repeats, "metrics": stability}

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MANIFEST_DIR / f"{phase}.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    return manifest
