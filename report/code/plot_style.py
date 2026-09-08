"""Shared publication-figure style for the SPIRAL2 LLRF report.

Every figure-generation script in report/code/ imports this module first and
calls apply_style() before creating any figure, so all figures share one
font, one line-width convention, one colorblind-safe palette, and one set of
physical widths matching the live report/main.tex (revtex4-2, reprint/aps/pra
= two-column). Widths were measured by compiling a minimal test document
against the real class/options, not assumed:
    single-column (\\linewidth)  = 246pt = 3.404 in
    full width    (\\textwidth, figure*) = 510pt = 7.058 in

Font: main.tex loads no Times/newtx package, so the document body font is
default Computer Modern. No CMU/Latin Modern font is installed in this
environment's matplotlib font cache (checked: only DejaVu and STIX are
available) -- STIXGeneral is used instead, as it was purpose-built to match
Computer/Times-style scientific typesetting and pairs natively with
matplotlib's 'stix' mathtext fontset, giving a coherent look without
requiring a local LaTeX compile (text.usetex stays False throughout, per
project convention -- Overleaf is the compile/render surface).

Color: the Okabe-Ito 8-color palette (Okabe & Ito, 2008) is used for every
categorical encoding -- it is the standard colorblind-safe qualitative
palette and remains distinguishable in grayscale print. Every categorical
series additionally gets a distinct marker (line plots) so color is never
the only channel. Sequential/diverging plots use perceptually-uniform,
colorblind-safe colormaps (cividis / PuOr), never jet or red-green.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Physical sizes (inches), measured against the live main.tex
# ---------------------------------------------------------------------------
SINGLE_COL_WIDTH = 3.404
FULL_WIDTH = 7.058

# ---------------------------------------------------------------------------
# Okabe-Ito colorblind-safe categorical palette
# ---------------------------------------------------------------------------
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
# Ordered list for cycling through categories (skips pure yellow first --
# poor contrast on white backgrounds -- reordered for typical use, yellow
# still available via OKABE_ITO["yellow"] when a distinguishing 8th color is
# genuinely needed, e.g. against a dark marker/hatch).
CATEGORICAL_ORDER = [
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["orange"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["black"],
    OKABE_ITO["yellow"],
]
# Distinct marker per series, same order as CATEGORICAL_ORDER, so color and
# shape stay paired across figures.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

SEQUENTIAL_CMAP = "cividis"
DIVERGING_CMAP = "PuOr"


def apply_style() -> None:
    """Set matplotlib rcParams shared by every figure in the report."""
    plt.rcParams.update(
        {
            # --- font ---
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 9,
            # --- lines / spines ---
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.1,
            "lines.markersize": 4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": matplotlib.cycler(color=CATEGORICAL_ORDER),
            # --- output ---
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # embed as real (searchable/editable) glyphs, not bitmaps
        }
    )


def series_style(i: int) -> dict:
    """Color/marker/linestyle kwargs for the i-th categorical series."""
    n = len(CATEGORICAL_ORDER)
    return {
        "color": CATEGORICAL_ORDER[i % n],
        "marker": MARKERS[i % n],
        "linestyle": LINESTYLES[i % n],
    }


def savefig_pdf(fig, path) -> None:
    fig.savefig(path, format="pdf")
    plt.close(fig)
