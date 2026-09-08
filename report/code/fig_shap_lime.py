#!/usr/bin/env python3
"""SHAP summary and LIME single-instance explanation figures
(fig:shap_summary, fig:lime_example -- 05_explainability.tex).

Real data / real models: reuses prepare_10_enhanced_shap_analysis.py's own
train_per_label_models() (one-vs-rest RandomForest per real fault category,
leakage-safe split) unchanged, rather than the Label Powerset SHAP archive
(shap_lp.npz), whose saved class axis (7 slices) doesn't line up 1:1 with its
own saved label_names (8 entries) -- an unresolved ambiguity not worth
building a labeled figure on. Per-label one-vs-rest models have an
unambiguous real fault-category label by construction, so both this SHAP
figure and the LIME figure below use that same, cleanly-labeled model family
for one representative, well-populated category (RF regulation out of
tolerance, n=591, the largest).

Must run under env_llrfspiral2 (sklearn/shap/lime versions installed there).
"""
import sys
from pathlib import Path

import numpy as np

PIPELINE_SCRIPTS = str(Path(__file__).resolve().parents[2] / "pipeline" / "00_scripts")
sys.path.insert(0, PIPELINE_SCRIPTS)
import os
os.environ.setdefault("SPIRAL2_COOKED_DIR", "/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6")

from prepare_10_enhanced_shap_analysis import load_data, train_per_label_models, FAULT_LABELS  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import SINGLE_COL_WIDTH, apply_style, savefig_pdf, DIVERGING_CMAP, OKABE_ITO  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
TARGET_CATEGORY = "RF regulation out of tolerance"
TOP_N = 12


def fig_shap_summary(model, X_test, feature_names, category):
    import matplotlib.pyplot as plt
    import shap

    max_samples = 500
    if len(X_test) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_test), max_samples, replace=False)
        X_explain = X_test[idx]
    else:
        X_explain = X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        # (n_samples, n_features, n_classes) -- take the positive class (index 1)
        shap_values = shap_values[:, :, 1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:TOP_N]

    apply_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 1.1))

    rng = np.random.RandomState(0)
    cmap = plt.get_cmap(DIVERGING_CMAP)
    for row, fi in enumerate(top_idx[::-1]):
        vals = shap_values[:, fi]
        fvals = X_explain[:, fi]
        # Normalize feature values to [0, 1] for color, robust to outliers (clip at 5/95 pct).
        lo, hi = np.percentile(fvals, [5, 95])
        norm = np.clip((fvals - lo) / (hi - lo + 1e-12), 0, 1)
        jitter = (rng.rand(len(vals)) - 0.5) * 0.6
        ax.scatter(vals, np.full_like(vals, row) + jitter, c=norm, cmap=cmap,
                   s=4, alpha=0.75, linewidths=0)

    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([feature_names[i] for i in top_idx[::-1]], fontsize=5.5)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("SHAP value (impact on model output)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title(category, fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / "shap_summary.pdf"
    savefig_pdf(fig, out)
    print(f"Wrote {out}")


def fig_lime_example(model, X_train, X_test, feature_names, category):
    import matplotlib.pyplot as plt
    from lime.lime_tabular import LimeTabularExplainer

    apply_style()

    explainer = LimeTabularExplainer(
        X_train, feature_names=feature_names, class_names=["Normal", category],
        discretize_continuous=True, random_state=42,
    )

    proba = model.predict_proba(X_test)[:, 1]
    # A confident true-positive instance -- the case an operator would actually see explained.
    candidates = np.where(proba > 0.9)[0]
    if len(candidates) == 0:
        candidates = np.argsort(-proba)[:1]
    inst_idx = candidates[0]

    exp = explainer.explain_instance(X_test[inst_idx], model.predict_proba, num_features=TOP_N)
    pairs = exp.as_list()
    labels = [p[0] for p in pairs][::-1]
    values = [p[1] for p in pairs][::-1]
    colors = [OKABE_ITO["vermillion"] if v > 0 else OKABE_ITO["blue"] for v in values]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.95))
    ax.barh(range(len(labels)), values, color=colors, height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.3)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("LIME local contribution")
    ax.set_title(f"{category}\n(model confidence: {proba[inst_idx]:.2f})", fontsize=6.8)
    fig.tight_layout()
    out = FIG_DIR / "lime_example.pdf"
    savefig_pdf(fig, out)
    print(f"Wrote {out}")


def main():
    pkl_data, y, feature_names = load_data()
    results = train_per_label_models(pkl_data, y, feature_names)
    if TARGET_CATEGORY not in results:
        raise SystemExit(f"{TARGET_CATEGORY} not in trained results: {list(results.keys())}")

    r = results[TARGET_CATEGORY]
    print(f"{TARGET_CATEGORY}: n_positive={r['n_positive']}, test_score={r['test_score']:.3f}")

    fig_shap_summary(r["model"], r["X_test"], feature_names, TARGET_CATEGORY)
    fig_lime_example(r["model"], r["X_train"], r["X_test"], feature_names, TARGET_CATEGORY)


if __name__ == "__main__":
    main()
