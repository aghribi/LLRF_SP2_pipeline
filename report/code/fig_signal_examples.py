#!/usr/bin/env python3
"""Preprocessed LLRF signal traces -- fig:signal_examples, 03_system_description.tex.

Real data: raw postmortem files, read via PyPostMortem.Read_Signals and run
through prepare_data_cluster_v6.py's own preprocess_signals() (high-pass
filter + Z-score + trigger alignment) unchanged -- the exact same code path
used to build the training features, not a reimplementation.

V6 FIX (2026-09-07, requested after review): the previous version plotted a
single "typical" (median-position) event per category, one line per panel.
Replaced with every real event in that category overlaid (thin, low-alpha
lines) plus the mean and a +-1 std band, matching the style of a legacy
exploratory figure (analysis/outputs/02_preprocessing_all_events.png, a much
smaller pre-V6 dataset, not reused directly) the author asked for -- a
single example can be atypical or cherry-picked-looking even when it
genuinely was picked at the category's median; showing every event answers
"is this representative?" directly rather than asking the reader to trust
that it is. Each event's own cropped window is resampled onto a common time
grid (events have different dt_us/sample counts) before averaging.

Each panel plots the channel that category's own real detection criterion
in Classify_PostMortemFile/config.yaml actually acts on (CHANNEL_BY_CATEGORY
below), not a single fixed channel for every panel -- an earlier version
always plotted Ucav, which made categories not driven by cavity-voltage
amplitude (e.g. Vacuum threshold, Pickup threshold) look like uninformative
near-copies of Normal.

Must run under env_llrfspiral2 (for PyPostmortem + consistent numpy/scipy).
Processes every event in every category (up to 1066 for Normal, 591 for RF
regulation out of tolerance) -- run as a batch job, not interactively.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/pbs/throng/m4cast/projects/SPIRAL2/programmes/PyPostMortem/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline" / "00_scripts"))
from prepare_data_cluster_v6 import preprocess_signals  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import FULL_WIDTH, apply_style, savefig_pdf, OKABE_ITO  # noqa: E402

V6_DIR = "/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

# Per-category channel: the signal each category's own real detection
# criterion actually acts on (Classify_PostMortemFile/config.yaml), not a
# single fixed channel for every panel -- showing Ucav for e.g. Vacuum
# threshold or Pickup threshold makes those panels look like near-copies
# of Normal, since Ucav isn't what triggers them. Grounded per category:
#   - Seuil pick-up: thresholds "courant pickup" (pickup current) directly.
#   - Coupure externe rapide: no dedicated monitored channel identified in
#     the classifier config (pass-through ALM-bit category, config.yaml's
#     classification.py match-case groups it with the other two
#     no-threshold-refinement categories) -- kept on Ucav as a general
#     operational-context default, not a claimed detection channel.
#   - Absence autorisation RF: RF drive presence/absence is what this
#     measures, and rf_drive_on (Sec. 4) is itself computed from Uci.
#   - Seuil de vide: thresholds "vide" (vacuum pressure) directly.
#   - Claquage ou quench cavité: config's quench threshold is a 50% Ucav
#     fall -- Ucav is the correct channel (voltage decay).
#   - Dép seuil de sécurité RF: config's rf_protection threshold is a
#     Ucav-based ratio -- Ucav is the correct channel.
#   - Rég signal RF hors tolérance: config's ecav_instability threshold is
#     a minimum-amplitude oscillation criterion on Ucav.
CHANNEL_BY_CATEGORY = {
    "Normal": "Ucav",
    "Pickup threshold": "courant pickup",
    "Fast external cutoff": "Ucav",
    "RF authorization absent": "Uci",
    "Vacuum threshold": "vide",
    "Cavity quench/breakdown": "Ucav",
    "RF safety threshold exceeded": "Ucav",
    "RF regulation out of tolerance": "Ucav",
}

# Display-only English label for each raw channel identifier (the raw names
# above are literal DAQ/dataframe column names -- some are French, e.g.
# "vide" (vacuum), "courant pickup" (pickup current) -- used for computation
# and for cross-referencing Appendix A's feature list, but not shown as
# panel-title text, which should read in English throughout.
CHANNEL_DISPLAY_NAME = {
    "Ucav": "Ucav",
    "Uci": "Uci",
    "courant pickup": "pickup current",
    "vide": "vacuum pressure",
}

# Known extreme-outlier events (near-zero-denominator / genuine rare transient
# events under separate investigation for the subtype-clustering figure) --
# excluded here so the example trace is representative, not a curated worst case.
OUTLIER_GLOBAL_IDX = {1962, 301, 687, 1550, 402, 664}

SHORT_NAMES = {
    "Seuil pick-up": "Pickup threshold",
    "Coupure externe rapide": "Fast external cutoff",
    "Absence autorisation RF": "RF authorization absent",
    "Seuil de vide": "Vacuum threshold",
    "Claquage ou quench cavité": "Cavity quench/breakdown",
    "Dép seuil de sécurité RF": "RF safety threshold exceeded",
    "Rég signal RF hors tolérance": "RF regulation out of tolerance",
}


def get_signals_and_zero_idx(path):
    from PyPostmortem.utils.PyPostMortem import Read_Signals

    with open(path, "rb") as f:
        content = f.read()
    parameters, time_arr, df_signals, _, _, _ = Read_Signals(
        content, compute_defauts=False, compute_etats=False, plot_signaux=False, show_header=False
    )
    metadata = {}
    for key, val_dict in parameters.items():
        if isinstance(val_dict, dict) and "Valeur" in val_dict:
            metadata[key] = val_dict["Valeur"]
        else:
            metadata[key] = val_dict
    signals = {col: df_signals[col].values for col in df_signals.columns}
    time_arr = np.asarray(time_arr)
    zero_idx = int(np.argmin(np.abs(time_arr)))
    dt_us = float(np.median(np.diff(time_arr))) if len(time_arr) > 1 else 1.0
    return signals, metadata, zero_idx, dt_us


# Common time grid every event's cropped window is resampled onto before
# overlay/averaging (events have different dt_us, hence different raw sample
# counts within the same +-34.07ms physical window).
GRID_N = 400
GRID_MS = np.linspace(-34.07, 34.07, GRID_N)

# Cap per category: a random (seeded, reproducible) subsample rather than
# every event -- for the largest categories (up to 1066/591) reading every
# single raw file is unnecessary compute, and overlaying that many vector
# lines bloats the PDF regardless of rasterization. 150 is enough for the
# mean/std band and the visual density/envelope to be stable; every
# category smaller than this is used in full (no sampling-down of already-
# small categories).
MAX_EVENTS_PER_CATEGORY = 150
SAMPLE_RNG = np.random.RandomState(0)


def gather_category_paths():
    with open(f"{V6_DIR}/features_engineered.pkl", "rb") as f:
        feat = pickle.load(f)
    with open(f"{V6_DIR}/processed_file_paths.pkl", "rb") as f:
        paths = pickle.load(f)

    y_multilabel = feat["y_multilabel"]
    y_binary = feat["y_binary"]
    fault_column_names = feat["fault_column_names"]

    def subsample(idxs):
        idxs = np.asarray(idxs)
        if len(idxs) > MAX_EVENTS_PER_CATEGORY:
            idxs = SAMPLE_RNG.choice(idxs, size=MAX_EVENTS_PER_CATEGORY, replace=False)
        return [paths[j] for j in idxs]

    groups = []
    normal_idx = [x for x in np.where(y_binary == 0)[0] if x not in OUTLIER_GLOBAL_IDX]
    groups.append(("Normal", subsample(normal_idx)))
    for i, name in enumerate(fault_column_names):
        idxs = [x for x in np.where(y_multilabel[:, i] == 1)[0] if x not in OUTLIER_GLOBAL_IDX]
        groups.append((SHORT_NAMES.get(name, name), subsample(idxs)))
    return groups


def load_one_event_multi(path, channels):
    """Load + preprocess once, return {channel: resampled trace or None} for
    every requested channel -- preprocess_signals() already computes every
    channel in one pass, so extracting several from the same call avoids
    re-reading/re-preprocessing the same raw file once per channel needed
    (Normal is reused as the comparison line on 4 different channels'
    panels; reading its ~1066 events 4x over would be wasteful)."""
    out = {c: None for c in channels}
    try:
        signals, metadata, zero_idx, dt_us = get_signals_and_zero_idx(path)
        processed = preprocess_signals(signals, metadata, zero_idx, dt_us)
        zoom_samples = int(round(34.07 * 1000 / dt_us))
        lo, hi = max(0, zero_idx - zoom_samples), min(len(processed["signals"].get(channels[0], [])), zero_idx + zoom_samples)
        for channel in channels:
            trace = processed["signals"].get(channel)
            if trace is None or hi <= lo or hi > len(trace):
                continue
            t_ms = (np.arange(lo, hi) - zero_idx) * dt_us / 1000.0
            out[channel] = np.interp(GRID_MS, t_ms, trace[lo:hi], left=np.nan, right=np.nan)
    except Exception as e:  # pragma: no cover - defensive only, reported not hidden
        print(f"WARNING: {path} failed: {e}")
    return out


def load_one_trace(path, channel):
    return load_one_event_multi(path, [channel])[channel]


def compute_normal_reference(normal_paths, channels):
    """Normal events are the comparison line on every fault panel, each on
    that panel's own channel -- Ucav (5 panels), courant pickup, Uci, vide
    (1 panel each). Read every Normal raw file exactly once, extracting all
    needed channels per event in that one pass (load_one_event_multi),
    rather than re-reading Normal's ~1066 events once per channel."""
    channels = sorted(set(channels))
    per_channel = {c: [] for c in channels}
    for p in normal_paths:
        traces = load_one_event_multi(p, channels)
        for c in channels:
            if traces[c] is not None:
                per_channel[c].append(traces[c])

    def mean_with_coverage_mask(vlist):
        a = np.vstack(vlist)
        m = np.nanmean(a, axis=0)
        n_valid = np.sum(np.isfinite(a), axis=0)
        min_valid = max(3, int(round(0.3 * len(vlist))))
        return np.where(n_valid < min_valid, np.nan, m)

    return {
        c: (mean_with_coverage_mask(v) if v else None)
        for c, v in per_channel.items()
    }


def main():
    import matplotlib.pyplot as plt

    apply_style()
    groups = gather_category_paths()
    groups_by_label = dict(groups)

    all_channels = sorted(set(CHANNEL_BY_CATEGORY.get(label, "Ucav") for label, _ in groups))
    print(f"Computing Normal reference means for channels: {all_channels}")
    normal_mean_by_channel = compute_normal_reference(groups_by_label["Normal"], all_channels)

    fig, axes = plt.subplots(2, 4, figsize=(FULL_WIDTH, 3.4), sharey=False)
    axes = axes.ravel()
    normal_handle = category_handle = None

    for ax, (label, paths) in zip(axes, groups):
        channel = CHANNEL_BY_CATEGORY.get(label, "Ucav")
        channel_display = CHANNEL_DISPLAY_NAME.get(channel, channel)
        traces = [load_one_trace(p, channel) for p in paths]
        traces = [t for t in traces if t is not None]
        n_loaded = len(traces)
        if n_loaded == 0:
            ax.text(0.5, 0.5, "no events loaded", ha="center", va="center", fontsize=5, transform=ax.transAxes)
            ax.set_title(f"{label}\n({channel_display}, n=0)", fontsize=6.5)
            ax.tick_params(labelsize=6)
            continue
        arr = np.vstack(traces)  # (n_loaded, GRID_N)
        color = OKABE_ITO["vermillion"] if label != "Normal" else OKABE_ITO["blue"]
        # Individual events: thin, low-alpha so density/envelope reads
        # visually rather than any single line dominating -- alpha scaled
        # down for the largest categories (up to 1066 events) so the plot
        # doesn't saturate to a solid block.
        alpha = float(np.clip(8.0 / n_loaded, 0.015, 0.3))
        # Rasterized (not left as vector paths): up to 150 lines per panel x
        # 8 panels as individual vector objects bloats the PDF and slows
        # every viewer/compiler that has to render them -- rasterizing just
        # this dense per-event layer (mean/std/axes/text stay vector, sharp
        # at any zoom) keeps file size and rendering cost bounded regardless
        # of how many events are overlaid. savefig.dpi=300 (plot_style.py)
        # sets the raster resolution.
        ax.plot(GRID_MS, arr.T, color=color, linewidth=0.4, alpha=alpha, rasterized=True)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        # Small categories can have events whose own cropped window doesn't
        # fully cover +-34.07ms (a shorter dt_us-scaled buffer near a file's
        # edge), leaving only a handful of events contributing at the grid's
        # extremes -- the mean/std there is then noise from n~1-2 events, not
        # a real population estimate. Mask (NaN, so matplotlib skips it)
        # any grid point backed by fewer than 30% of this category's loaded
        # events (floor of 3), rather than plot an unstably-estimated tail.
        n_valid = np.sum(np.isfinite(arr), axis=0)
        min_valid = max(3, int(round(0.3 * n_loaded)))
        low_coverage = n_valid < min_valid
        mean = np.where(low_coverage, np.nan, mean)
        std = np.where(low_coverage, np.nan, std)
        (h,) = ax.plot(GRID_MS, mean, color=color, linewidth=1.1, zorder=5)
        ax.fill_between(GRID_MS, mean - std, mean + std, color=color, alpha=0.2, linewidth=0, zorder=4)
        if label != "Normal":
            category_handle = h
            # Bold Normal reference on the SAME channel this panel uses, for
            # direct visual contrast rather than requiring the reader to
            # flip back to the separate Normal panel.
            normal_mean = normal_mean_by_channel.get(channel)
            if normal_mean is not None:
                (normal_handle,) = ax.plot(
                    GRID_MS, normal_mean, color=OKABE_ITO["blue"], linewidth=1.1,
                    linestyle="--", zorder=6,
                )
        ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
        # Robust y-limits over the full pooled (all events, all times) array,
        # not raw min/max -- a handful of outlier events would otherwise
        # compress everything else, the same problem a plain min/max axis
        # has with a single event's own post-trigger transient.
        finite = arr[np.isfinite(arr)]
        if finite.size:
            p_lo, p_hi = np.percentile(finite, [1, 99])
            pad = 0.15 * max(p_hi - p_lo, 1e-9)
            ax.set_ylim(p_lo - pad, p_hi + pad)
        ax.set_title(f"{label}\n({channel_display}, n={n_loaded})", fontsize=6.5)
        ax.tick_params(labelsize=6)
        print(f"{label}: {n_loaded}/{len(paths)} events loaded")

    for ax in axes[4:]:
        ax.set_xlabel("Time rel. trigger (ms)", fontsize=6.5)
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("Filtered signal (z-score)", fontsize=6.5)

    if normal_handle is not None and category_handle is not None:
        fig.legend(
            [normal_handle, category_handle], ["Normal (mean, same channel)", "This category (mean $\\pm$1 std)"],
            loc="upper center", ncol=2, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.04),
        )

    fig.tight_layout()
    out = FIG_DIR / "signal_examples.pdf"
    savefig_pdf(fig, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
