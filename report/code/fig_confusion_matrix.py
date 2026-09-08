#!/usr/bin/env python3
"""Root-cause identification confusion matrix (fig:phase08_confusion, 06_results.tex).

Real data: cooked_data_v6/step_08_phase3a_v6/root_cause.pkl, the reference
single split (seed=42) already produced this session -- y_test, predictions,
fault_names. No retraining, no rerun.

Must run under env_llrfspiral2 (sklearn 1.6.1) to unpickle the RandomForestClassifier.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import SINGLE_COL_WIDTH, SEQUENTIAL_CMAP, apply_style, savefig_pdf

V6_DIR = "/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6"
OUT = Path(__file__).resolve().parents[1] / "figures" / "root_cause_confusion_matrix.pdf"

# Short labels for axis ticks -- full names appear in the caption / Table 1 (03_system_description).
SHORT_NAMES = {
    "Seuil pick-up": "Pickup\nthreshold",
    "Coupure externe rapide": "Fast ext.\ncutoff",
    "Absence autorisation RF": "RF auth.\nabsent",
    "Seuil de vide": "Vacuum\nthreshold",
    "Claquage ou quench cavité": "Cavity\nquench",
    "Dép seuil de sécurité RF": "RF safety\nthreshold",
    "Rég signal RF hors tolérance": "RF reg.\nout of tol.",
}


def main():
    import matplotlib.pyplot as plt

    apply_style()

    with open(f"{V6_DIR}/step_08_phase3a_v6/root_cause.pkl", "rb") as f:
        d = pickle.load(f)

    y_test = d["data_splits"]["y_test"]
    y_pred = d["predictions"]
    fault_names = d["fault_names"]
    labels = list(range(len(fault_names)))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    short = [SHORT_NAMES[n] for n in fault_names]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.92))
    im = ax.imshow(cm_norm, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticklabels(short)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Annotate raw counts; text color flips for readability against the colormap.
    for i in range(len(short)):
        for j in range(len(short)):
            val = cm[i, j]
            if val == 0:
                continue
            frac = cm_norm[i, j]
            color = "white" if frac > 0.55 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=6.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized fraction", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    n_test = len(y_test)
    ax.set_title(f"Reference split, n={n_test} test events", fontsize=7.5)

    fig.tight_layout()
    savefig_pdf(fig, OUT)
    print(f"Wrote {OUT}")
    print("Row order / short-label mapping:")
    for full, s in zip(fault_names, short):
        print(f"  {full!r} -> {s!r}")


if __name__ == "__main__":
    main()
