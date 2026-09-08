#!/usr/bin/env python3
"""Early fault-onset detection (step 05) figures: score distributions, ROC
curves, and feature importance -- fig:phase04_scores, fig:phase05_roc,
fig:phase05_importance in 04_methodology.tex.

Real data: cooked_data_v6/step_05_v6/precursor_detection.pkl (this session's
V6 run) -- per-event scores for every unsupervised method, the trained RF's
probabilities, y_test, and real feature_importance. No rerun.

Must run under env_llrfspiral2 (sklearn 1.6.1) to unpickle the models.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import SINGLE_COL_WIDTH, apply_style, savefig_pdf, series_style, OKABE_ITO

V6_DIR = "/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

METHOD_LABELS = {
    "isolation_forest": "Isolation Forest",
    "lof": "LOF",
    "mahalanobis": "Mahalanobis",
    "pca_reconstruction": "PCA reconstruction",
}


def load():
    with open(f"{V6_DIR}/step_05_v6/precursor_detection.pkl", "rb") as f:
        return pickle.load(f)


def _ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def fig_score_distributions(d):
    import matplotlib.pyplot as plt

    y_test = d["data_splits"]["y_test"]
    methods = ["isolation_forest", "lof", "mahalanobis", "pca_reconstruction"]

    # Empirical CDFs rather than histograms: these anomaly scores are extremely
    # heavy-tailed (a handful of outliers span orders of magnitude beyond the
    # bulk), so a binned/linear-axis histogram either hides the bulk behind the
    # outliers or clips them away. ECDFs on a log-scaled x-axis show the full
    # separation between classes without either problem.
    fig, axes = plt.subplots(2, 2, figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.95), sharey=True)
    axes = axes.ravel()

    for ax, m in zip(axes, methods):
        scores = d[m]["scores"]
        auc_val = d[m]["roc_auc"]
        normal = scores[y_test == 0]
        fault = scores[y_test == 1]
        shift = 0.0
        if scores.min() <= 0:
            shift = -scores.min() + 1e-6 * (scores.max() - scores.min() + 1e-9)
        xs_n, ys_n = _ecdf(normal + shift)
        xs_f, ys_f = _ecdf(fault + shift)
        ax.plot(xs_n, ys_n, color=OKABE_ITO["blue"], label="Normal", linewidth=1.1)
        ax.plot(xs_f, ys_f, color=OKABE_ITO["vermillion"], label="Fault", linewidth=1.1, linestyle="--")
        ax.set_xscale("log")
        ax.set_title(f"{METHOD_LABELS[m]} (AUC={auc_val:.2f})", fontsize=7)
        ax.tick_params(labelsize=6)

    axes[0].legend(fontsize=6, loc="lower right", frameon=False)
    axes[0].set_ylabel("Cumulative fraction", fontsize=7)
    axes[2].set_ylabel("Cumulative fraction", fontsize=7)
    for ax in axes[2:]:
        ax.set_xlabel("Anomaly score (shifted, log scale)", fontsize=6.5)
    fig.tight_layout()
    out = FIG_DIR / "precursor_unsupervised_scores.pdf"
    savefig_pdf(fig, out)
    print(f"Wrote {out}")


def fig_roc(d):
    import matplotlib.pyplot as plt

    y_test = d["data_splits"]["y_test"]

    series = [
        ("Random Forest (supervised)", d["random_forest"]["probabilities"], d["random_forest"]["roc_auc"]),
        ("Ensemble (unsupervised + RF)", d["ensemble"]["scores"], d["ensemble"]["roc_auc"]),
        ("Isolation Forest (unsupervised)", d["isolation_forest"]["scores"], d["isolation_forest"]["roc_auc"]),
    ]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.92))
    for i, (label, scores, auc_val) in enumerate(series):
        fpr, tpr, _ = roc_curve(y_test, scores)
        style = series_style(i)
        # Sparse markers: too many points on a near-perfect curve is unreadable.
        marker_every = max(1, len(fpr) // 12)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})",
                color=style["color"], linestyle=style["linestyle"],
                marker=style["marker"], markevery=marker_every, markersize=3.5)
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.8, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=6, loc="lower right", frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "precursor_roc_curves.pdf"
    savefig_pdf(fig, out)
    print(f"Wrote {out}")


def fig_importance(d, top_n=15):
    import matplotlib.pyplot as plt

    fi = d["random_forest"]["feature_importance"].sort_values("importance", ascending=False).head(top_n)
    fi = fi.iloc[::-1]  # smallest at bottom for a horizontal bar chart read top-to-bottom by rank

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 1.05))
    ax.barh(fi["feature"], fi["importance"], color=OKABE_ITO["blue"], height=0.65)
    ax.set_xlabel("RF feature importance (Gini)")
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    out = FIG_DIR / "precursor_feature_importance.pdf"
    savefig_pdf(fig, out)
    print(f"Wrote {out}")


def main():
    apply_style()
    d = load()
    fig_score_distributions(d)
    fig_roc(d)
    fig_importance(d)


if __name__ == "__main__":
    main()
