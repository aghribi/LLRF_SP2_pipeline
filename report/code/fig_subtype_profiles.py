#!/usr/bin/env python3
"""Fault subtype clustering feature profiles (fig:phase09_profiles, 04_methodology.tex).

Real data: cooked_data_v6/step_09c_enhanced_subclass/enhanced_subclass_full.pkl
(per_fault_clustering: best_labels per event, in the same order that
get_fault_mask(y_multilabel, fault_idx) selects them, per
prepare_09c_enhanced_subclass_discovery.py) combined post-hoc with
features_engineered.pkl (X_scaled, 766 real named features, same row order as
y_multilabel) to compute real per-subtype feature-mean profiles. No
retraining -- best_labels are reused exactly as produced by that script; this
script only re-selects the same rows to average named features by cluster.

Must run under env_llrfspiral2 (sklearn 1.6.1) to unpickle.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import FULL_WIDTH, apply_style, savefig_pdf, OKABE_ITO

V6_DIR = "/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
TOP_N = 6

SHORT_NAMES = {
    "RF authorization absent": "RF authorization absent (n=309)",
    "Vacuum threshold": "Vacuum threshold (n=20)",
    "Cavity quench/breakdown": "Cavity quench/breakdown (n=28)",
    "RF safety threshold exceeded": "RF safety threshold exceeded (n=72)",
    "RF regulation out of tolerance": "RF regulation out of tolerance (n=591)",
}

# prepare_09c_enhanced_subclass_discovery.py's FAULT_LABELS (English display
# names) is positionally aligned with features_engineered.pkl's
# fault_column_names (French, canonical order) -- both are the same 7
# categories in the same order, just different display strings. Map by
# position, not name (the strings genuinely differ), verified against the
# source script.
FAULT_LABELS_EN = [
    "Pickup threshold",
    "Fast external cutoff",
    "RF authorization absent",
    "Vacuum threshold",
    "Cavity quench/breakdown",
    "RF safety threshold exceeded",
    "RF regulation out of tolerance",
]


def get_fault_mask(y, fault_idx):
    return y[:, fault_idx] == 1


def main():
    import matplotlib.pyplot as plt

    apply_style()

    with open(f"{V6_DIR}/features_engineered.pkl", "rb") as f:
        feat = pickle.load(f)
    with open(f"{V6_DIR}/step_09c_enhanced_subclass/enhanced_subclass_full.pkl", "rb") as f:
        sub = pickle.load(f)

    y_multilabel = feat["y_multilabel"]
    X_scaled = feat["X_scaled"]
    feature_cols = feat["feature_cols"]
    fault_column_names = feat["fault_column_names"]

    per_fault = sub["per_fault_clustering"]

    fig, axes = plt.subplots(1, len(per_fault), figsize=(FULL_WIDTH, 3.3))

    for ax, entry in zip(axes, per_fault):
        fault_name = entry["fault_name"]
        best_labels = np.asarray(entry["best_labels"])
        fault_idx = FAULT_LABELS_EN.index(fault_name)
        mask = get_fault_mask(y_multilabel, fault_idx)
        X_fault = X_scaled[mask]
        # Real cross-check, not just a positional assumption: mask.sum() must equal
        # both the saved n_samples and len(best_labels) for this to be a valid pairing.
        assert X_fault.shape[0] == best_labels.shape[0] == entry["n_samples"], (
            f"{fault_name} (fault_idx={fault_idx}, French name={fault_column_names[fault_idx]!r}): "
            f"mask gave {X_fault.shape[0]} rows, best_labels has {best_labels.shape[0]}, "
            f"manifest says n_samples={entry['n_samples']}"
        )
        print(f"OK: {fault_name!r} -> fault_idx={fault_idx} ({fault_column_names[fault_idx]!r}), n={entry['n_samples']}")

        # Winsorize at the 5th/95th percentile WITHIN this category before computing
        # effect sizes -- the same treatment cluster_within_fault_type() applies before
        # clustering (prepare_09c_enhanced_subclass_discovery.py). Without it, a single
        # remaining extreme value in one of the smaller clusters (n as low as 4) can still
        # collapse that group's variance for one feature and blow up its Cohen's d.
        X_fault_w = np.clip(X_fault, np.percentile(X_fault, 5, axis=0), np.percentile(X_fault, 95, axis=0))
        g0 = X_fault_w[best_labels == 0]
        g1 = X_fault_w[best_labels == 1]
        mean_diff = g1.mean(axis=0) - g0.mean(axis=0)
        # Cohen's d (pooled within-category std): a bounded effect size, robust
        # to the fact that a handful of features have huge global-scaler variance
        # within this small category subset -- a raw z-score mean difference is
        # unbounded and was dominated by scaling artifacts, not real separation.
        n0, n1 = len(g0), len(g1)
        pooled_std = np.sqrt(((n0 - 1) * g0.var(axis=0, ddof=1) + (n1 - 1) * g1.var(axis=0, ddof=1))
                              / max(n0 + n1 - 2, 1))
        pooled_std = np.where(pooled_std < 1e-9, np.nan, pooled_std)
        cohens_d = mean_diff / pooled_std
        cohens_d = np.nan_to_num(cohens_d, nan=0.0)
        # A handful of features are still ~constant within one of the two (now
        # balanced but not huge) subgroups -- e.g. an integer-valued feature like
        # peak_count sharing the same value across an entire small subgroup --
        # which drives pooled_std to near-zero and d into the thousands for that
        # one feature alone. Clip to +-5: by Cohen's conventional interpretation
        # (0.2/0.5/0.8 = small/medium/large), anything beyond 5 is already an
        # enormous effect and not meaningfully different from "extreme" for
        # ranking/display purposes -- this bounds the display without hiding
        # which features are most discriminative.
        cohens_d = np.clip(cohens_d, -5, 5)

        top_idx = np.argsort(-np.abs(cohens_d))[:TOP_N]
        top_idx = top_idx[np.argsort(cohens_d[top_idx])]  # ascending, for a clean diverging bar order

        labels = [_short_feature_name(feature_cols[i]) for i in top_idx]
        values = cohens_d[top_idx]
        colors = [OKABE_ITO["vermillion"] if v > 0 else OKABE_ITO["blue"] for v in values]

        ax.barh(range(len(labels)), values, color=colors, height=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=5.5)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(SHORT_NAMES.get(fault_name, fault_name), fontsize=6.3, wrap=True)
        ax.tick_params(axis="x", labelsize=6)
        vmax = max(abs(values.min()), abs(values.max()), 0.1) * 1.2
        ax.set_xlim(-vmax, vmax)

    fig.supxlabel("Subtype 1 vs. subtype 0 effect size (Cohen's d, within-category)", fontsize=7)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.86, bottom=0.14, wspace=0.75)
    out = FIG_DIR / "subtype_feature_profiles.pdf"
    savefig_pdf(fig, out)
    print(f"Wrote {out}")


def _short_feature_name(name: str, max_len: int = 26) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


if __name__ == "__main__":
    main()
