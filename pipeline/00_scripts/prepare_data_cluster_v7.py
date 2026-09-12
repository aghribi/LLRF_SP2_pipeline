#!/usr/bin/env python3
"""
LLRF Anomaly Detection - Data Preparation for Cluster (V7 - feed-forward filter/feature fix)
================================================================================

Forked from prepare_data_cluster_v6.py (2026-09-09). V6 stays as-is (it's a
real, documented checkpoint); this is the new authoritative extraction path.

FIX (2026-09-09) -- V7, on top of V6:
- The ground-truth classifier's dominant quality-filter reason (68% of every
  rejection, previously logged as "Beam NON") was found to be a mislabeled
  check: the raw header field named BEAM (values OUI/NON) does not encode
  particle-beam presence -- it encodes whether feed-forward is enabled in the
  LLRF control loop, unrelated to beam presence. The field is also only
  operationally meaningful from 2021 onward (before that it defaults to NON on
  ~100% of files, since feed-forward wasn't deployed yet, not a real signal).
  Classify_PostMortemFile's _check_filters() has been fixed to scope this
  check to acquisitions >= 2021 and the ground-truth CSV (LLRF_result.csv)
  re-run against the fix -- see report/sections/03_system_description.tex for
  the full writeup and Classify_PostMortemFile/config.yaml's
  feed_forward_min_year. This recovers on the order of several hundred
  legitimate 2019-2020 events that the unscoped check was silently discarding.
- The same mislabeling existed in this script's own feature matrix: the
  'beam_present' feature (from metadata['BEAM']) is renamed 'feed_forward_enabled'
  to match its real meaning. Values are unchanged (same underlying header
  field/logic), only the name and its documentation. Any downstream script
  that hardcodes the old feature name ('beam_present') for grouping/taxonomy
  purposes (e.g. prepare_10_enhanced_shap_analysis.py's physics-group mapping,
  reaggregate_features.py) needs the same rename to keep picking it up --
  checked and fixed as part of this same change where those scripts exist.
- No other extraction logic changes from V6 (trigger alignment, AMPT onset
  refinement, whole-signal-statistic normalization-order fix, interlock_type
  removal are all unchanged, inherited as-is).
================================================================================

Forked from prepare_data_cluster_v5.py (2026-09-04 leakage investigation). V5
stays as-is (it's a real, documented checkpoint); this is the new authoritative
extraction path.

FIX (2026-09-04) -- V6, on top of V5:
- Removed the `interlock_type` feature (ALM-bit-derived: -1 if no ALM interlock,
  else the lowest set ALM bit index). It is computed from the exact same `ALM`
  metadata field that V2's label (`y_binary = alm > 0`) was computed from, and
  was `PROTECTED_PHYSICS_FEATURES`-exempted from correlation filtering. On V2
  this made it a perfect single-feature predictor of the label (interlock_type
  == -1 iff y_binary == 0) -- confirmed as the root cause of XGBoost's literal
  100%-on-every-metric result flagged in NOTEBOOK_AUDIT_2026-09-03.md. Even
  after V3+ switched labels to the validated ground-truth source (breaking the
  perfect equivalence -- ALM and ground truth disagree ~38% of the time),
  interlock_type remained a real, moderate leak on V5 (naive single-threshold
  accuracy 76.4%, corr 0.53 against V5's y_binary) since ALM state is
  contemporaneous with/downstream of the fault event itself. Dropped outright
  rather than kept-with-caveat: it doesn't carry information a legitimate
  pre-fault-window feature couldn't, and its presence would keep inflating
  binary/multi-label classification metrics for any model expressive enough to
  find it. All other V5 logic (ground-truth labels, trigger alignment, AMPT
  onset refinement) is unchanged.

FIX (2026-09-07, found during external review), on top of the above:
- engineer_features() computed each signal's whole-signal mean/std/rms/
  energy from the SAME array whose own mean/std were used to Z-score-
  normalize it in preprocess_signals() -- so mean~=0 and std~=rms~=1 for
  every event by mathematical construction, and energy was mostly a proxy
  for record length (NDEC-dependent), not real signal energy. Verified
  directly against the previous features_engineered.pkl: 65 of 770 final
  features were near-constant (std < 1e-6) across all 2109 events as a
  result, including this bug and two related ones below. Two absolute-
  unit threshold features had the same root problem: rf_drive_on's
  threshold (1.0) is calibrated against real Uci amplitude (RF-on
  ~1.5-1.7, RF-off ~0.000001) and was always 0 once applied to Z-scored
  data instead; reflected_saturation_fraction's threshold (95.0) is of
  unknown provenance and may still not correspond to a real saturation
  level for this signal even in real units -- fixed to use real-amplitude
  data structurally, but not recalibrated, since we have no grounds to
  invent a new threshold value.
  Fix: preprocess_signals() now also returns 'signals_unscaled' (filtered
  but not Z-score-normalized); engineer_features() uses it for these
  specific features instead of the Z-scored 'signals'. Every other
  statistical feature (skewness, kurtosis, extrema, quartiles, crest
  factor, peak count, zero crossings, autocorrelation) is scale/shift-
  invariant or uses an already-relative threshold and is unaffected, so
  intentionally left on the Z-scored signal.

--- V5's docstring follows (still accurate for everything except interlock_type) ---

Forked from prepare_data_cluster_v4.py per the 2026-09-04 cleanup plan, Phase 2
Layer 2. V4 stays as-is (it's a real, documented checkpoint); this is the new
authoritative extraction path.

FIXES (2026-09-04) -- V5, on top of V4's corrected informatic-trigger alignment:
- V4 fixed the *informatic* trigger position (zero_idx, from Read_Signals' time
  axis). But Alexandre Dalibard--Brun's report (Rapport_Stage_Alexandre.pdf,
  section 4.6.1) documents that the informatic trigger doesn't always coincide
  with the true PHYSICAL onset of the anomaly -- decoding latency and hardware
  propagation delays mean the real onset can precede or follow it. His
  Classify_PostMortemFile repo implements find_true_t0_with_ampt(), which uses
  the file's own AMPT (amplitude-tolerance) header parameter to detect the real
  voltage-drop/rise onset.
- Scoped empirically before implementing (300-file random sample of V4's
  accepted events): 76.3% of files show ZERO drift (find_true_t0_with_ampt
  short-circuits when AMPT<=3%, i.e. most events); of the 23.7% that do drift,
  the effect correlates directly with AMPT (differing files average AMPT~9.5%,
  vs 4.2% for non-differing files) and the magnitude is usually tiny (median
  ~0.01ms) but reaches up to ~2.1ms in the tail -- non-trivial against some of
  the smaller bounded windows (e.g. the ~5.7ms decay window).
- Per user decision: applied ONLY to events whose primary category is "Rég
  signal RF hors tolérance" -- the one category Alexandre actually validated
  this refinement against, and the one his own thesis work targeted it for.
  All other categories keep V4's plain informatic zero_idx unchanged.
- `y_alignment_refined` (bool array) added to the output pickle so it's always
  possible to tell, per event, whether the AMPT refinement fired.

--- V4's docstring follows (still accurate: informatic trigger alignment,
    superseded by nothing -- V5 only adds a further refinement on top) ---

Forked from prepare_data_cluster_v3.py per the 2026-09-04 cleanup plan, Phase 2
Layer 1. V3 stays as-is (it's a real, documented checkpoint with its own dataset
numbers); this is the new authoritative extraction path.

FIXES (2026-09-04) -- V4, on top of V3's ground-truth labeling:
- CRITICAL: every physics/precursor feature computation assumed the informatic
  trigger sits at a hardcoded sample index 3000 (`trigger_idx`/`delta_pre = 3000`,
  ~20 call sites). Verified empirically (172-file random sample, all years,
  all NDEC values 20-5000) that this is essentially never true: `Read_Signals`
  already returns a correct per-file time axis (accounting for NDEC and
  POSTROW) with the real trigger at `argmin(|time|)`, but this script discarded
  that axis immediately (`.values`) and never used it. For the DOMINANT case
  (NDEC=200, symmetric buffer, ~87% of sampled files) the true trigger index is
  ~50047, not 3000. POSTROW/NROW ranges from 0% to 99.9% across the corpus;
  NROW itself ranges from ~62k to ~2.1M samples; NDEC from 20 to 5000 (a ~250x
  range in sample spacing). All of 2025 sampled at ~95% pre-trigger, confirming
  the acquisition reconfiguration mentioned in Rapport_Stage_Alexandre.pdf.
- Fix: `zero_idx` (true trigger sample index) and `dt_us` (this file's actual
  sample spacing) are now computed once per file from the preserved time axis,
  and threaded through preprocess_signals()/engineer_features(). Two distinct
  window semantics, both now correct:
    (a) "steady-state" pre/post features (detuning, RF-mismatch, modulator,
        RF-power-flow means) use ALL available pre-trigger / post-trigger data,
        split at the correct `zero_idx` -- matches what the code's own
        "steady-state" comments already described as intent.
    (b) "precursor" / bounded-duration features (preprocess_signals' 5 segments,
        engineer_features section 4's trend/CUSUM/early-late-window features,
        the small windows around calculate_effective_decay and
        power_jump_at_trigger) use a FIXED PHYSICAL DURATION taken immediately
        before/after `zero_idx`, converted to a per-file sample count via that
        file's own `dt_us` -- so every file gets the same window in
        milliseconds, not the same (physically meaningless, given NDEC varies
        250x) window in samples. Reference durations are chosen to match V2/V3's
        dominant-case (NDEC=200) numeric behavior, so results stay comparable
        for the ~87% of files where the old code happened to use a
        physically-similar-duration window, even though it was positioned wrong.
- `fs = 88000` (hardcoded, only correct for NDEC=200) replaced with
  `fs = 1e6 / dt_us`, computed per file.
- Segment boundaries' misleading stale comments ("T-170ms to T-136ms" etc, which
  never matched the actual code even before this fix) removed rather than
  preserved -- see PRECURSOR_WINDOW_MS below for the actual duration used.

--- V3's docstring follows (still accurate: ground-truth labeling, unrelated to
    this file's trigger-alignment fix) ---

Forked from prepare_data_cluster.py (V2) per the 2026-09-04 cleanup plan, Phase 1.
Do not edit prepare_data_cluster.py to match this file -- V2 stays as-is for
provenance/comparison; this is the new authoritative extraction path.

FIXES (2026-09-04) -- V3, on top of V2's MEMORY-OPTIMIZED baseline:
- CRITICAL: Replaced ALM-header-only labeling (extract_fault_labels) with a
  ground-truth lookup against Classify_PostMortemFile's deterministic,
  physics-based classifier (see LABEL_SOURCE below). The ALM header field was
  found unreliable (Lassalle thesis: 38.33% disagreement vs. waveform-level
  fault register on a 2,497-file sample; independently reproduced in-house).
- The ground truth's own filtering (BEAM/LOOP/KPI/MASK/AMPT/Ucav-healthy-
  duration -- strictly more complete than this script's old KPI/Ucav/Uci-only
  passes_quality_filters) is now the SOLE quality gate. passes_quality_filters
  is no longer called: a file is included iff it has a ground-truth row whose
  Label is not "Filtré". This is a superset check done by filename BEFORE the
  raw file is even opened, so excluded files cost near-zero time.
- multilabel category order is unchanged from V2's ALM-bit order (index i
  means the same fault type in both), now derived from the ground truth's
  single `Defaut` category as a one-hot vector instead of noisy ALM bits.
  fault_column_names now holds the real category names instead of 'Fault_i'.
  A new y_physics_subtype array carries the finer-grained physics label
  (e.g. "Quench", "Oscillation Ucav") for the RF-regulation category.
- Consequence: dataset size changes again (this is a tightening, matching the
  7,427->4,509 precedent) -- expect fewer events than V2's 4,509, since the
  ground truth's filter is stricter and files without a valid ground-truth
  row (previously: unclassified faults; now: also true no-fault "Normal"
  events, processing errors, non-postmortem stray files) are excluded rather
  than silently mislabeled.

LABEL_SOURCE = programmes/Classify_PostMortemFile/result/LLRF_result.csv
  (cloned from https://gitlab.in2p3.fr/22002019/Classify_PostMortemFile,
  Alexandre Dalibard--Brun's internship work; run 2026-09-04 with all 7 fault
  types + an added "Normal" class enabled, full 2019-2025 raw corpus.)

Everything else (feature engineering, PCA, batching/checkpointing, memory
optimizations) is unchanged from V2 -- see that file's own docstring below
for the original MEMORY-OPTIMIZED V2 changelog.

--- Original V2 docstring follows ---

Optimized for CC-IN2P3 SLURM cluster with:
- Per-file timeout protection (prevents infinite hangs)
- Multi-processing support
- Checkpoint/resume capability
- Better error logging
- MEMORY-EFFICIENT MODE: Don't store full signals in batch files

FIXES (2026-01-25):
- CRITICAL: Reduced memory footprint by not storing full signals in batch pickles
- Batch files now only contain: features, metadata, labels, file paths
- Sequence arrays built in a separate memory-mapped pass if needed
- Reduced default batch size from 2000 to 500
- Added explicit garbage collection after each batch
- This fixes OOM errors when processing large datasets (14k+ files)

Usage:
    # Multi-processing (recommended) - memory efficient by default
    python prepare_data_cluster.py --data-dir /path/to/data --output-dir /path/to/output --n-jobs 4

    # Resume from last batch (after job was killed)
    python prepare_data_cluster.py --data-dir /path/to/data --output-dir /path/to/output --resume

    # Store full signals (NOT recommended - high memory, only for small datasets)
    python prepare_data_cluster.py --data-dir /path/to/data --output-dir /path/to/output --store-signals

    # Custom timeout for slow files
    python prepare_data_cluster.py --data-dir /path/to/data --output-dir /path/to/output --file-timeout 60
"""

import argparse
import pickle
import logging
import re
import sys
import time
import signal
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count, TimeoutError as MP_TimeoutError
from functools import partial
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Ground-truth label source (V3) -- see module docstring for full rationale.
# =============================================================================
GROUND_TRUTH_CSV = Path(
    '/pbs/throng/m4cast/projects/SPIRAL2/programmes/Classify_PostMortemFile/result/LLRF_result.csv'
)

# Same order as V2's ALM-bit multilabel scheme -- index i is the same fault
# type in both, so downstream consumers of fault_column_names/y_multilabel
# that assumed the old bit order keep working.
FAULT_CATEGORIES = [
    'Seuil pick-up',
    'Coupure externe rapide',
    'Absence autorisation RF',
    'Seuil de vide',
    'Claquage ou quench cavité',
    'Dép seuil de sécurité RF',
    'Rég signal RF hors tolérance',
]

# =============================================================================
# Trigger alignment (V4) -- see module docstring for full rationale.
# =============================================================================
# Sample spacing for NDEC=200, the dominant case (~87% of a 172-file random
# sample across all years). NOT used as a fallback for missing data -- every
# file's own dt_us (measured from its own time axis) is what actually gates
# processing. Used only to convert the ORIGINAL hardcoded sample-count windows
# (which happened to be correct-duration, if wrongly positioned, for this
# dominant case) into physical durations, so V4 stays numerically comparable to
# V2/V3 for the common case while being correctly positioned for every case.
REFERENCE_DT_US = 11.356860963627696

# "Precursor" / bounded-duration window: the original code's implicit 3000-
# sample pre-trigger window, expressed in ms via the reference dt above.
PRECURSOR_WINDOW_MS = 3000 * REFERENCE_DT_US / 1000  # ~34.07 ms


def scale_samples(n_reference_samples, dt_us):
    """
    Convert a sample count that was originally tuned against REFERENCE_DT_US
    into the equivalent sample count for a file with its own dt_us, preserving
    physical duration. Used for every bounded/precursor window in this file.
    """
    if not dt_us or dt_us <= 0:
        dt_us = REFERENCE_DT_US
    return max(1, int(round(n_reference_samples * REFERENCE_DT_US / dt_us)))


# =============================================================================
# Physical-onset refinement (V5) -- see module docstring for full rationale.
# Only applied to "Rég signal RF hors tolérance" events, per user decision.
# =============================================================================
CLASSIFY_POSTMORTEM_SRC = Path('/pbs/throng/m4cast/projects/SPIRAL2/programmes/Classify_PostMortemFile/src')
ONSET_REFINEMENT_CATEGORY = 'Rég signal RF hors tolérance'
_find_true_t0_with_ampt = None  # lazy import, see refine_trigger_index()


def refine_trigger_index(ucav_array, zero_idx, ampt, defaut):
    """
    Returns the alignment index to actually use for this event: the informatic
    trigger (zero_idx) unchanged, UNLESS this event is the one category
    (ONSET_REFINEMENT_CATEGORY) Alexandre validated find_true_t0_with_ampt
    against, in which case returns his AMPT-refined physical onset instead.

    Returns (aligned_idx, was_refined: bool).
    """
    if defaut != ONSET_REFINEMENT_CATEGORY:
        return zero_idx, False

    global _find_true_t0_with_ampt
    if _find_true_t0_with_ampt is None:
        src = str(CLASSIFY_POSTMORTEM_SRC)
        if src not in sys.path:
            sys.path.insert(0, src)
        from signal_processing import find_true_t0_with_ampt
        _find_true_t0_with_ampt = find_true_t0_with_ampt

    try:
        ucav_series = pd.Series(np.asarray(ucav_array))
        refined_idx, _ = _find_true_t0_with_ampt(ucav_series, zero_idx, ampt)
        refined_idx = int(refined_idx)
        return refined_idx, (refined_idx != zero_idx)
    except Exception:
        # Defensive fallback: never let a refinement failure break extraction.
        return zero_idx, False


_GROUND_TRUTH_CACHE = None


def load_ground_truth():
    """
    Load and cache the Classify_PostMortemFile ground-truth CSV as a dict
    keyed by raw filename (no path, no extension games -- the CSV's
    `Fichier` column already matches raw post-mortem filenames exactly).

    Cached at module level so each worker process (re-imported under
    multiprocessing) loads it once, not once per file.
    """
    global _GROUND_TRUTH_CACHE
    if _GROUND_TRUTH_CACHE is not None:
        return _GROUND_TRUTH_CACHE

    df = pd.read_csv(GROUND_TRUTH_CSV, sep=';')
    lookup = {}
    for row in df.itertuples(index=False):
        lookup[row.Fichier] = {
            'defaut': row.Defaut,
            'label': row.Label if isinstance(row.Label, str) else '',
            'details': row.Détails if isinstance(row.Détails, str) else '',
        }
    _GROUND_TRUTH_CACHE = lookup
    logger.info(f"Loaded ground truth: {len(lookup)} rows from {GROUND_TRUTH_CSV}")
    return lookup


def extract_fault_labels_from_ground_truth(file_path):
    """
    Look up (file_path's basename) in the ground truth. Returns None if the
    file has no usable ground-truth row (not present -- processing error,
    zero-byte, or a non-postmortem stray file; or present but Label=="Filtré"
    -- failed the ground truth's own, stricter quality gate). A None return
    means "skip this file" -- the caller must not process it further.

    Otherwise returns the same {'binary', 'multilabel', ...} shape V2's
    extract_fault_labels produced, so downstream code (save_results) needs
    no changes, plus two new keys: 'defaut' (category name) and
    'physics_subtype' (finer-grained label, populated only for the RF-
    regulation category; None otherwise).
    """
    ground_truth = load_ground_truth()
    row = ground_truth.get(Path(file_path).name)
    if row is None:
        return None
    if row['label'] == 'Filtré':
        return None

    defaut = row['defaut']
    has_fault = defaut != 'Normal'
    multilabel = [0] * len(FAULT_CATEGORIES)
    if has_fault and defaut in FAULT_CATEGORIES:
        multilabel[FAULT_CATEGORIES.index(defaut)] = 1

    physics_subtype = row['label'] if defaut == 'Rég signal RF hors tolérance' else None

    return {
        'binary': int(has_fault),
        'multilabel': multilabel,
        'defaut': defaut,
        'physics_subtype': physics_subtype,
    }


class TimeoutException(Exception):
    """Exception raised when file processing times out"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("File processing timeout")


def process_single_file_with_timeout(file_path, file_timeout=30, store_signals=False):
    """
    Process a single LLRF data file with timeout protection

    Args:
        file_path: Path to file
        file_timeout: Timeout in seconds (default 30s)
        store_signals: If True, return full signals (high memory). Default False.

    Returns:
        dict with processing results or error info
    """
    # V3: ground-truth gate FIRST, by filename, before opening/decoding the
    # raw file at all. Cheap for the ~85% of files this excludes (Filtré,
    # no ground-truth row).
    fault_labels = extract_fault_labels_from_ground_truth(file_path)
    if fault_labels is None:
        return None

    # Set up timeout (UNIX only)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(file_timeout)

    try:
        # Import here to avoid issues with multiprocessing
        try:
            # Add PyPostMortem src directory to path if not already there
            pypm_path = Path('/pbs/throng/m4cast/projects/SPIRAL2/programmes/PyPostMortem/src')
            pypm_path_str = str(pypm_path.resolve())

            if pypm_path_str not in sys.path:
                sys.path.insert(0, pypm_path_str)

            from PyPostmortem.utils.PyPostMortem import Read_Signals
        except ImportError as ie:
            error_msg = (
                f"Missing required module 'PyPostmortem': {ie}\n"
                f"Please ensure PyPostMortem is available at: {pypm_path_str}"
            )
            return {'file': str(file_path), 'success': False, 'error': error_msg}

        # Read signals and parameters (metadata)
        with open(file_path, 'rb') as f:
            file_content = f.read()

        # Read_Signals returns 6 values: parameters, time, df_signaux, df_defaut, df_states, Header
        parameters, time_arr, df_signals, _, _, _ = Read_Signals(
            file_content,
            compute_defauts=False,
            compute_etats=False,
            plot_signaux=False,
            show_header=False
        )

        # Cancel timeout
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

        # Extract metadata from parameters
        metadata = {}
        for key, val_dict in parameters.items():
            if isinstance(val_dict, dict) and 'Valeur' in val_dict:
                metadata[key] = val_dict['Valeur']
            else:
                metadata[key] = val_dict

        # Convert DataFrame columns to dict of arrays for signals
        signals = {col: df_signals[col].values for col in df_signals.columns}

        # V4: the true informatic trigger index and this file's own sample
        # spacing, both read from Read_Signals' own time axis (see module
        # docstring) instead of the V2/V3 hardcoded trigger_idx=3000.
        time_arr = np.asarray(time_arr)
        zero_idx = int(np.argmin(np.abs(time_arr)))
        dt_us = float(np.median(np.diff(time_arr))) if len(time_arr) > 1 else REFERENCE_DT_US

        # V3: quality filtering and label extraction already done above via
        # the ground truth (Classify_PostMortemFile) -- passes_quality_filters
        # and extract_fault_labels (V2's ALM-based versions) are intentionally
        # not called here. `fault_labels` was already resolved by the early
        # gate near the top of this function.

        # V5: refine zero_idx to the true physical onset, but only for the one
        # category Alexandre validated this against (see module docstring).
        # All other categories keep V4's plain informatic zero_idx.
        ampt = float(metadata.get('AMPT', 0) or 0)
        aligned_idx, alignment_refined = refine_trigger_index(
            signals.get('Ucav'), zero_idx, ampt, fault_labels['defaut']
        )
        fault_labels['alignment_refined'] = alignment_refined

        # Preprocess signals
        signals_processed = preprocess_signals(signals, metadata, aligned_idx, dt_us)

        # Engineer features
        features = engineer_features(signals_processed, metadata, aligned_idx, dt_us)

        # Get signal names for reference (lightweight - just names, not data)
        signal_names = list(signals_processed['signals'].keys())

        # MEMORY OPTIMIZATION: By default, don't return full signals
        # This reduces memory from ~2.6MB/event to ~50KB/event
        if store_signals:
            return {
                'file': str(file_path),
                'metadata': metadata,
                'signals': signals_processed,  # Full signals (high memory)
                'features': features,
                'fault_labels': fault_labels,
                'signal_names': signal_names,
                'success': True
            }
        else:
            # Lightweight mode - only return features, metadata, labels
            return {
                'file': str(file_path),
                'metadata': metadata,
                'features': features,
                'fault_labels': fault_labels,
                'signal_names': signal_names,
                'success': True
            }

    except TimeoutException:
        # Cancel timeout
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        error_msg = f"TIMEOUT after {file_timeout}s"
        logger.error(f"Error processing {file_path}: {error_msg}")
        return {'file': str(file_path), 'success': False, 'error': error_msg}

    except Exception as e:
        # Cancel timeout
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        logger.error(f"Error processing {file_path}: {e}")
        return {'file': str(file_path), 'success': False, 'error': str(e)}


# NOTE (V3): passes_quality_filters and extract_fault_labels below are V2's
# original ALM-based functions, kept for reference/comparison only -- NOT
# called anywhere in this V3 file. See extract_fault_labels_from_ground_truth
# above for what actually gates and labels events now.
def passes_quality_filters(metadata, signals):
    """
    Check if event passes quality filters

    IMPORTANT: We need BOTH fault and no-fault events for ML training!
    - Fault events: ALM > 0 (automatic triggers)
    - Normal events: ALM = 0 (manual acquisitions, no faults)

    CRITICAL FIX: Removed NDEC filter that was rejecting most data!
    Previously only accepted NDEC=200, which filtered out ~60-80% of files.
    Now accepts ALL NDEC values to maximize training data.

    NEW: Added RF power filters to ensure cavities are turned on:
    - mean(Ucav) > 0.1 (cavity voltage amplitude)
    - mean(Uci) > 0.02 (incident/forward signal amplitude)

    Based on empirical data from SPIRAL2:
    - RF ON: Ucav ~ 3-4, Uci ~ 1.5-1.7
    - RF OFF: Ucav ~ 0.002-0.003, Uci ~ 0.000001

    Args:
        metadata: Dict with event metadata (KPI, ALM, etc.)
        signals: Dict of raw signal arrays (before preprocessing)

    Returns:
        bool: True if event passes all filters
    """
    quality_filters = {
        'min_kpi': 10.0,
        'require_fault': False,  # ✅ Include both fault and normal events
        'min_ucav': 0.1,  # Minimum cavity voltage - ensures RF is on
        'min_uci': 0.02,  # Minimum forward voltage - ensures RF drive is present
    }

    # KPI filter (minimum quality threshold)
    if 'KPI' in metadata:
        try:
            kpi = float(metadata['KPI'])
            if kpi < quality_filters['min_kpi']:
                return False
        except (ValueError, TypeError):
            return False

    # =========================================================================
    # RF Power Filters - ensure cavities are turned on (NEW)
    # =========================================================================

    # Filter 1: Cavity voltage (Ucav) must be above threshold
    # This ensures the cavity is actually powered on
    # Typical values: RF ON ~3-4, RF OFF ~0.002-0.003
    if 'Ucav' in signals:
        try:
            ucav = np.array(signals['Ucav'])
            ucav_mean = np.mean(np.abs(ucav))

            if ucav_mean < quality_filters['min_ucav']:
                return False  # Cavity voltage too low - RF likely off
        except (ValueError, TypeError, AttributeError):
            return False
    else:
        # If Ucav signal is missing, reject the event
        return False

    # Filter 2: Forward/incident voltage (Uci) must be above threshold
    # This ensures RF drive is present at the cavity input
    # Typical values: RF ON ~1.5-1.7, RF OFF ~0.000001
    if 'Uci' in signals:
        try:
            uci = np.array(signals['Uci'])
            uci_mean = np.mean(np.abs(uci))

            if uci_mean < quality_filters['min_uci']:
                return False  # Forward voltage too low - no RF drive
        except (ValueError, TypeError, AttributeError):
            return False
    else:
        # If Uci signal is missing, reject the event
        return False

    # NDEC filter - REMOVED to accept all NDEC values
    # Previously this was rejecting most files by only accepting NDEC=200:
    #
    # if 'NDEC' in metadata:
    #     try:
    #         ndec = int(metadata['NDEC'])
    #         if ndec not in [200]:  # ← This rejected ~60-80% of data!
    #             return False
    #     except (ValueError, TypeError):
    #         return False
    #
    # Now we accept ALL NDEC values to maximize training data.
    # If specific NDEC filtering is needed, it can be added later.

    return True


def extract_fault_labels(metadata):
    """Extract fault labels from metadata"""
    alm_raw = metadata.get('ALM', 0)
    try:
        if isinstance(alm_raw, str):
            alm = int(alm_raw, 16) if alm_raw else 0
        else:
            alm = int(alm_raw) if alm_raw else 0
    except (ValueError, TypeError):
        alm = 0

    has_fault = alm > 0

    # Multi-label fault types (7 bits)
    fault_bits = []
    for i in range(7):
        fault_bits.append(int((alm >> i) & 1))

    return {
        'binary': int(has_fault),
        'multilabel': fault_bits,
        'alm_value': alm
    }


def preprocess_signals(signals, metadata, zero_idx, dt_us):
    """
    Preprocess signals: filtering, normalization, derivatives, segmentation

    Args:
        signals: dict of raw signal arrays (positional indexing -- the pandas
            time index from Read_Signals was already stripped by the caller)
        metadata: dict with ALM, LOOP, KPI, etc.
        zero_idx: this file's true trigger sample index (V4: from Read_Signals'
            own time axis, NOT a hardcoded position -- see module docstring)
        dt_us: this file's own sample spacing in microseconds

    Returns: dict with keys:
        - 'signals': normalized signals
        - 'signals_unscaled': filtered but NOT Z-score-normalized signals
          (V6 fix, see below) -- same high-pass filtering as 'signals', just
          without the per-signal amplitude normalization
        - 'first_deriv': 1st derivatives (velocity)
        - 'second_deriv': 2nd derivatives (acceleration)
        - 'segments': 5 temporal segments of the fixed-duration precursor
          window immediately before zero_idx (PRECURSOR_WINDOW_MS, ~34.07ms --
          same physical duration for every file regardless of NDEC)

    V6 FIX (2026-09-07, found during external review): engineer_features()
    previously computed each signal's whole-signal mean/std/rms/energy (and
    two absolute-threshold features, reflected_saturation_fraction and
    rf_drive_on) from 'signals' -- the SAME array whose own mean/std were
    just used to Z-score it here. That makes those features close to a
    mathematical constant (mean~=0, std~=rms~=1 for every event, to float
    precision) rather than real per-event descriptors, and makes any fixed
    absolute-unit threshold applied to that array physically meaningless.
    'signals_unscaled' is exposed so engineer_features() can compute those
    specific features on real, un-normalized amplitude information instead.
    """
    from scipy import signal as sp_signal

    processed = {
        'signals': {},
        'signals_unscaled': {},
        'first_deriv': {},
        'second_deriv': {},
        'segments': {}
    }

    # V4: bounded precursor window, immediately before the true trigger,
    # scaled to this file's own dt_us so every file gets the same PHYSICAL
    # duration (V2/V3 used a fixed sample count, which meant a duration that
    # varied ~250x across the corpus depending on NDEC).
    precursor_window_samples = scale_samples(3000, dt_us)
    seg_len = precursor_window_samples // 5
    segment_boundaries = {
        'early': (0, seg_len),
        'mid_early': (seg_len, 2 * seg_len),
        'mid': (2 * seg_len, 3 * seg_len),
        'mid_late': (3 * seg_len, 4 * seg_len),
        'late': (4 * seg_len, precursor_window_samples),
    }

    for signal_name, signal_data in signals.items():
        data = np.array(signal_data)

        # High-pass filter (remove DC offset)
        if len(data) > 10:
            sos = sp_signal.butter(4, 0.01, 'highpass', output='sos')
            data = sp_signal.sosfilt(sos, data)

        # V6 fix: keep the filtered-but-unnormalized signal before Z-scoring
        # destroys its real amplitude information (see docstring above).
        processed['signals_unscaled'][signal_name] = data.copy()

        # Normalization (z-score)
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            data = (data - mean) / std

        processed['signals'][signal_name] = data

        # Compute derivatives
        first_deriv = np.gradient(data)
        second_deriv = np.gradient(first_deriv)

        processed['first_deriv'][signal_name] = first_deriv
        processed['second_deriv'][signal_name] = second_deriv

        # Extract temporal segments: the precursor window immediately before
        # the true trigger (zero_idx), not "the first N samples of the file"
        # (V2/V3's bug -- for most files that was ~500ms+ before the trigger).
        window_start = max(0, zero_idx - precursor_window_samples)
        pre_trigger_data = data[window_start:zero_idx]
        if len(pre_trigger_data) >= seg_len:
            processed['segments'][signal_name] = {}
            for segment_name, (start, end) in segment_boundaries.items():
                if end <= len(pre_trigger_data):
                    processed['segments'][signal_name][segment_name] = pre_trigger_data[start:end]

    return processed


def engineer_features(signals_processed, metadata, zero_idx, dt_us):
    """
    Engineer comprehensive features from preprocessed signals

    Extracts 1200+ features (ALL signals):
    - Statistical features (17 per signal × 27 signals = 459)
      Includes: mean, std, min, max, range, median, q1, q3, iqr,
               skewness, kurtosis, rms, crest_factor, peak_count,
               energy, n_zero_crossings, autocorr_lag1
    - Physics features (44 total - CORRECTED PER THESIS TABLE 2.2):
      * Effective decay: 5 features (NOT Q_L - invalid for CW with closed loops)
      * Detuning: 5 features (mean, std, slope, peak-to-peak, jump at trigger)
      * Microphonics: 4 features (RMS, dominant freq, band power, Q-factor)
      * RF mismatch: 7 features (reflected-amplifier difference pre/post, excursion, settling, saturation)
        NOTE: A Ucr = reflected signal amplitude, A Uamp = amplifier voltage output (NOT V_ref)
      * Modulator command: 6 features (IQ amplitude/phase pre, changes post, max)
        These represent actual control effort (I/Q modulateur from thesis Table 2.2)
      * Phase stability: 4 features (jitter RMS, excursion, slope, PSD integral)
      * RF power: 5 features (forward/reflected mean, ratio, jump, imbalance)
      * Validity flags: 5 features (RF on, loop closed, feed-forward enabled, interlock type, valid decay)
      * Legacy: 3 features (kept for backward compatibility)
    - Precursor features (~729): trends, CUSUM, derivatives, segments, variance ratios
    - Metadata features (~10)

    Signal definitions per Charly Lassalle thesis Table 2.2:
      - A Ucr: Amplitude of REFLECTED signal (column 9)
      - A Uamp: Amplifier voltage output (column 10)
      - I/Q modulateur: Modulator command I/Q components (columns 11-12)
      - IQ: Modulator command magnitude = sqrt(I² + Q²)
      - MODP: Modulator command phase = arctan2(Q, I)

    Args:
        signals_processed: dict with keys 'signals', 'first_deriv', 'second_deriv', 'segments'
        metadata: dict with ALM, LOOP, KPI, etc.
        zero_idx: this file's true trigger sample index (V4 -- see module docstring)
        dt_us: this file's own sample spacing in microseconds

    Returns:
        dict of features
    """
    from scipy import stats as sp_stats
    from scipy.signal import find_peaks

    features = {}

    # Extract components
    signals = signals_processed['signals']
    signals_unscaled = signals_processed['signals_unscaled']
    first_deriv = signals_processed['first_deriv']
    second_deriv = signals_processed['second_deriv']
    segments = signals_processed['segments']

    # ========================================================================
    # 1. STATISTICAL FEATURES (17 per signal × 27 signals = 459 features)
    # ========================================================================
    for signal_name, signal_data in signals.items():
        # V6 fix: mean/std/rms/energy use the filtered-but-UNSCALED signal --
        # computing them on the Z-scored 'signal_data' (as before) makes them
        # close to a mathematical constant (mean~=0, std~=rms~=1 for every
        # event), since that's exactly the array whose own mean/std were used
        # to normalize it (see preprocess_signals() docstring). Every other
        # statistic below is scale/shift-invariant or uses a relative
        # threshold, so it's unaffected and stays on the Z-scored signal.
        unscaled_data = signals_unscaled[signal_name]

        # Basic statistics
        features[f'{signal_name}_mean'] = np.mean(unscaled_data)
        features[f'{signal_name}_std'] = np.std(unscaled_data)
        features[f'{signal_name}_min'] = np.min(signal_data)
        features[f'{signal_name}_max'] = np.max(signal_data)
        features[f'{signal_name}_range'] = np.ptp(signal_data)
        features[f'{signal_name}_median'] = np.median(signal_data)

        # Quartiles
        features[f'{signal_name}_q1'] = np.percentile(signal_data, 25)
        features[f'{signal_name}_q3'] = np.percentile(signal_data, 75)
        features[f'{signal_name}_iqr'] = features[f'{signal_name}_q3'] - features[f'{signal_name}_q1']

        # Distribution shape
        features[f'{signal_name}_skewness'] = sp_stats.skew(signal_data)
        features[f'{signal_name}_kurtosis'] = sp_stats.kurtosis(signal_data)

        # RMS (unscaled, same fix as mean/std/energy above) and crest factor
        # (scale-invariant by construction, so it keeps using the Z-scored
        # max/rms pair -- mixing an unscaled numerator with a Z-scored
        # denominator would make it meaningless).
        features[f'{signal_name}_rms'] = np.sqrt(np.mean(unscaled_data**2))
        zscore_rms = np.sqrt(np.mean(signal_data**2))
        crest = features[f'{signal_name}_max'] / zscore_rms if zscore_rms > 0 else 0
        features[f'{signal_name}_crest_factor'] = crest

        # Peak analysis
        peaks, properties = find_peaks(signal_data, prominence=0.5)
        features[f'{signal_name}_peak_count'] = len(peaks)

        # Energy (sum of squared values, unscaled -- same fix as above)
        features[f'{signal_name}_energy'] = np.sum(unscaled_data**2)

        # Zero crossings (sign changes)
        zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
        features[f'{signal_name}_n_zero_crossings'] = len(zero_crossings)

        # Autocorrelation at lag 1
        if len(signal_data) > 1:
            features[f'{signal_name}_autocorr_lag1'] = np.corrcoef(signal_data[:-1], signal_data[1:])[0, 1]
        else:
            features[f'{signal_name}_autocorr_lag1'] = 0.0

    # ========================================================================
    # 2. PHYSICS FEATURES (38 features total - UPDATED)
    # ========================================================================
    # Sampling frequency: V4 -- per-file, from this file's own dt_us (was a
    # hardcoded 88000 Hz, only correct for the dominant NDEC=200 case).
    fs = 1e6 / dt_us

    # Calculate effective system decay metrics (NOT Q_L)
    # In CW operation with closed loops, this is NOT intrinsic Q_L!
    if 'Ucav' in signals:
        (features['effective_decay_tau'],
         features['decay_rate'],
         features['decay_fit_r2'],
         features['decay_relative_to_nominal'],
         features['decay_non_exponentiality']) = calculate_effective_decay(
            signals['Ucav'], zero_idx, fs=fs, decay_window_samples=scale_samples(500, dt_us)
        )
    else:
        features['effective_decay_tau'] = np.nan
        features['decay_rate'] = np.nan
        features['decay_fit_r2'] = 0.0
        features['decay_relative_to_nominal'] = np.nan
        features['decay_non_exponentiality'] = np.nan

    # Effective detuning from phase drift (5 features)
    if 'PhaseCav' in signals:
        phase_cav = signals['PhaseCav']
        trigger_idx = zero_idx  # V4: real trigger (was hardcoded 3000)

        # Pre-trigger analysis for detuning
        phase_pre = phase_cav[:trigger_idx] if len(phase_cav) > trigger_idx else phase_cav
        phase_post = phase_cav[trigger_idx:] if len(phase_cav) > trigger_idx else []

        # Unwrap phase
        phase_unwrapped = np.unwrap(phase_pre)

        # Time vector
        t_pre = np.arange(len(phase_pre)) / fs

        # Linear fit: φ(t) = φ0 + 2π*Δf*t
        if len(phase_unwrapped) > 100:
            coeffs = np.polyfit(t_pre, phase_unwrapped, 1)
            phase_slope = coeffs[0]  # rad/s

            # Convert to detuning in Hz
            detuning_hz = phase_slope / (2 * np.pi)

            # Detuning features
            features['detuning_mean_hz'] = detuning_hz

            # Std of instantaneous detuning (from phase derivative)
            dt = t_pre[1] - t_pre[0] if len(t_pre) > 1 else 1/fs
            phase_deriv = np.gradient(phase_unwrapped) / dt  # rad/s
            detuning_inst = phase_deriv / (2 * np.pi)
            features['detuning_std_hz'] = np.std(detuning_inst)

            # Detuning drift rate (second derivative)
            phase_accel = np.gradient(phase_deriv) / dt  # rad/s²
            features['detuning_slope_hz_s'] = np.mean(phase_accel) / (2 * np.pi)

            # Peak-to-peak excursion
            features['detuning_peak_to_peak'] = (np.max(phase_unwrapped) - np.min(phase_unwrapped)) / (2 * np.pi)

            # Jump at trigger
            if len(phase_post) > 10:
                phase_post_unwrapped = np.unwrap(phase_post)
                jump = (phase_post_unwrapped[10] - phase_unwrapped[-1]) / (2 * np.pi)
                features['detuning_jump_at_trigger'] = jump
            else:
                features['detuning_jump_at_trigger'] = 0.0
        else:
            features['detuning_mean_hz'] = 0.0
            features['detuning_std_hz'] = 0.0
            features['detuning_slope_hz_s'] = 0.0
            features['detuning_peak_to_peak'] = 0.0
            features['detuning_jump_at_trigger'] = 0.0
    else:
        features['detuning_mean_hz'] = 0.0
        features['detuning_std_hz'] = 0.0
        features['detuning_slope_hz_s'] = 0.0
        features['detuning_peak_to_peak'] = 0.0
        features['detuning_jump_at_trigger'] = 0.0

    # ========================================================================
    # RF POWER MISMATCH FEATURES (7 features)
    # NOTE: These are NOT control errors. They quantify the difference between
    # reflected signal amplitude (A Ucr) and amplifier output voltage (A Uamp).
    # Signal definitions per thesis Table 2.2:
    #   - A Ucr: Amplitude of REFLECTED signal (not reference)
    #   - A Uamp: Amplifier voltage output
    # ========================================================================
    if 'A Ucr' in signals and 'A Uamp' in signals and 'PhaseCav' in signals:
        # Power mismatch: reflected - amplifier output (NOT a control error)
        rf_mismatch = signals['A Ucr'] - signals['A Uamp']
        phase_cav = signals['PhaseCav']

        trigger_idx = zero_idx  # V4: real trigger (was hardcoded 3000)

        # Pre-trigger (steady-state)
        mismatch_pre = rf_mismatch[:trigger_idx] if len(rf_mismatch) > trigger_idx else rf_mismatch
        phase_pre = phase_cav[:trigger_idx] if len(phase_cav) > trigger_idx else phase_cav

        features['rf_mismatch_rms_pre'] = np.sqrt(np.mean(mismatch_pre**2))
        features['phase_std_pre'] = np.std(phase_pre)

        # Post-trigger (transient)
        if len(rf_mismatch) > trigger_idx:
            mismatch_post = rf_mismatch[trigger_idx:]
            phase_post = phase_cav[trigger_idx:]

            features['rf_mismatch_rms_post'] = np.sqrt(np.mean(mismatch_post**2))
            features['phase_std_post'] = np.std(phase_post)

            # Max excursion (max in first 500 samples post-trigger)
            excursion_window = min(500, len(mismatch_post))
            features['rf_mismatch_max_excursion'] = np.max(np.abs(mismatch_post[:excursion_window]))

            # Settling time (±5% of final value)
            try:
                final_val = np.median(mismatch_post[-100:])
                threshold = 0.05 * np.abs(final_val) if final_val != 0 else 0.05
                settled = np.where(np.abs(mismatch_post - final_val) < threshold)[0]
                features['rf_mismatch_settling_time'] = settled[0] / fs if len(settled) > 0 else np.inf
            except:
                features['rf_mismatch_settling_time'] = np.nan

            # Reflected power saturation detection. V6 fix: use the
            # unscaled (real-amplitude) signal, not the Z-scored one, since
            # a fixed absolute threshold is meaningless in Z-scored units
            # (see preprocess_signals() docstring); saturation_level itself
            # is an unverified legacy constant we did not have grounds to
            # recalibrate, so this feature may still read as near-always-zero
            # if 95.0 does not correspond to a real saturation level for
            # this signal's physical units -- left as a known open question
            # (Appendix A) rather than guessing a new threshold.
            ucr = signals_unscaled['A Ucr']
            saturation_level = 95.0
            saturated_samples = np.sum(np.abs(ucr) > saturation_level)
            features['reflected_saturation_fraction'] = saturated_samples / len(ucr)
        else:
            features['rf_mismatch_rms_post'] = np.nan
            features['phase_std_post'] = np.nan
            features['rf_mismatch_max_excursion'] = 0.0
            features['rf_mismatch_settling_time'] = np.nan
            features['reflected_saturation_fraction'] = 0.0
    else:
        features['rf_mismatch_rms_pre'] = 0.0
        features['phase_std_pre'] = 0.0
        features['rf_mismatch_rms_post'] = np.nan
        features['phase_std_post'] = np.nan
        features['rf_mismatch_max_excursion'] = 0.0
        features['rf_mismatch_settling_time'] = np.nan
        features['reflected_saturation_fraction'] = 0.0

    # ========================================================================
    # CONTROL LOOP FEATURES - ALIASES (7 features)
    # These provide the exact feature names expected by analysis notebooks
    # while maintaining compatibility with existing rf_mismatch features
    # ========================================================================
    # Amplitude RMS (control error proxy) - alias for rf_mismatch_rms
    features['amp_rms_pre'] = features.get('rf_mismatch_rms_pre', 0.0)
    features['amp_rms_post'] = features.get('rf_mismatch_rms_post', np.nan)

    # Phase RMS (control error) - alias for phase_std
    features['phase_rms_pre'] = features.get('phase_std_pre', 0.0)
    features['phase_rms_post'] = features.get('phase_std_post', np.nan)

    # Control overshoot - max excursion in first 500 samples post-trigger
    features['control_overshoot_amp'] = features.get('rf_mismatch_max_excursion', 0.0)

    # Control settling time - alias for rf_mismatch_settling_time
    features['control_settling_time'] = features.get('rf_mismatch_settling_time', np.nan)

    # Control saturation fraction - alias for reflected_saturation_fraction
    features['control_saturation_fraction'] = features.get('reflected_saturation_fraction', 0.0)

    # ========================================================================
    # MODULATOR COMMAND FEATURES (6 features)
    # These represent the actual control effort sent to the RF amplifier.
    # Signal definitions per thesis Table 2.2:
    #   - I modulateur: In-phase component of modulator command
    #   - Q modulateur: Quadrature component of modulator command
    #   - IQ: Magnitude of modulator command = sqrt(I² + Q²)
    #   - MODP: Phase of modulator command = arctan2(Q, I)
    # ========================================================================
    if 'IQ' in signals and 'MODP' in signals:
        mod_amplitude = signals['IQ']
        mod_phase = signals['MODP']
        trigger_idx = zero_idx  # V4: real trigger (was hardcoded 3000)

        # Pre-trigger modulator statistics
        mod_amp_pre = mod_amplitude[:trigger_idx] if len(mod_amplitude) > trigger_idx else mod_amplitude
        mod_phase_pre = mod_phase[:trigger_idx] if len(mod_phase) > trigger_idx else mod_phase

        features['mod_amplitude_mean_pre'] = np.mean(mod_amp_pre)
        features['mod_amplitude_std_pre'] = np.std(mod_amp_pre)
        features['mod_phase_std_pre'] = np.std(mod_phase_pre)

        # Post-trigger modulator response
        if len(mod_amplitude) > trigger_idx:
            mod_amp_post = mod_amplitude[trigger_idx:]
            mod_phase_post = mod_phase[trigger_idx:]

            features['mod_amplitude_change'] = np.mean(mod_amp_post) - np.mean(mod_amp_pre)
            features['mod_phase_change'] = np.mean(mod_phase_post) - np.mean(mod_phase_pre)
            features['mod_amplitude_max_post'] = np.max(np.abs(mod_amp_post))
        else:
            features['mod_amplitude_change'] = 0.0
            features['mod_phase_change'] = 0.0
            features['mod_amplitude_max_post'] = 0.0
    else:
        features['mod_amplitude_mean_pre'] = 0.0
        features['mod_amplitude_std_pre'] = 0.0
        features['mod_phase_std_pre'] = 0.0
        features['mod_amplitude_change'] = 0.0
        features['mod_phase_change'] = 0.0
        features['mod_amplitude_max_post'] = 0.0

    # Phase Stability Features (4 features)
    if 'PhaseCav' in signals:
        phase_cav_stab = signals['PhaseCav']
        trigger_idx = zero_idx  # V4: real trigger (was hardcoded 3000)
        phase_pre_stab = phase_cav_stab[:trigger_idx] if len(phase_cav_stab) > trigger_idx else phase_cav_stab

        # RMS jitter
        features['phase_jitter_rms'] = np.std(phase_pre_stab)

        # Peak-to-peak excursion
        features['phase_excursion_pp'] = np.ptp(phase_pre_stab)

        # Low-frequency drift (linear slope)
        if len(phase_pre_stab) > 100:
            t_stab = np.arange(len(phase_pre_stab)) / fs
            phase_unwrapped_stab = np.unwrap(phase_pre_stab)
            try:
                coeffs_stab = np.polyfit(t_stab, phase_unwrapped_stab, 1)
                features['phase_noise_slope'] = coeffs_stab[0]  # rad/s
            except:
                features['phase_noise_slope'] = 0.0
        else:
            features['phase_noise_slope'] = 0.0

        # Integrated PSD
        try:
            freqs_psd = np.fft.rfftfreq(len(phase_pre_stab), 1/fs)
            fft_mag_psd = np.abs(np.fft.rfft(phase_pre_stab))
            psd = fft_mag_psd**2 / len(phase_pre_stab)
            features['phase_psd_integral'] = np.sum(psd)
        except:
            features['phase_psd_integral'] = 0.0
    else:
        features['phase_jitter_rms'] = 0.0
        features['phase_excursion_pp'] = 0.0
        features['phase_noise_slope'] = 0.0
        features['phase_psd_integral'] = 0.0

    # Microphonics Spectral Features (4 features)
    if 'PhaseCav' in signals:
        phase_cav_micro = signals['PhaseCav']
        trigger_idx = zero_idx  # V4: real trigger (was hardcoded 3000)
        phase_pre_micro = phase_cav_micro[:trigger_idx] if len(phase_cav_micro) > trigger_idx else phase_cav_micro

        if len(phase_pre_micro) > 100:
            # Residual phase (remove linear trend)
            phase_unwrapped_micro = np.unwrap(phase_pre_micro)
            coeffs_micro = np.polyfit(np.arange(len(phase_unwrapped_micro)), phase_unwrapped_micro, 1)
            phase_trend = np.polyval(coeffs_micro, np.arange(len(phase_unwrapped_micro)))
            phase_residual = phase_unwrapped_micro - phase_trend

            # RMS microphonics
            features['microphonics_rms'] = np.std(phase_residual)

            # FFT analysis
            try:
                freqs_micro = np.fft.rfftfreq(len(phase_residual), 1/fs)
                fft_mag_micro = np.abs(np.fft.rfft(phase_residual))

                # Dominant frequency
                if len(fft_mag_micro) > 1:
                    dom_idx = np.argmax(fft_mag_micro[1:]) + 1  # Skip DC
                    features['microphonics_dom_freq'] = freqs_micro[dom_idx]

                    # Band power (10-100 Hz typical for mechanical vibrations)
                    band_mask = (freqs_micro >= 10) & (freqs_micro <= 100)
                    features['microphonics_band_power_10_100'] = np.sum(fft_mag_micro[band_mask]**2)

                    # Resonance Q-factor (sharpness of dominant peak)
                    # Q = f_peak / FWHM
                    peak_mag = fft_mag_micro[dom_idx]
                    half_max = peak_mag / 2

                    # Find FWHM
                    left_idx = dom_idx
                    while left_idx > 0 and fft_mag_micro[left_idx] > half_max:
                        left_idx -= 1
                    right_idx = dom_idx
                    while right_idx < len(fft_mag_micro)-1 and fft_mag_micro[right_idx] > half_max:
                        right_idx += 1

                    fwhm = freqs_micro[right_idx] - freqs_micro[left_idx]
                    features['microphonics_q_factor'] = freqs_micro[dom_idx] / fwhm if fwhm > 0 else 0
                else:
                    features['microphonics_dom_freq'] = 0.0
                    features['microphonics_band_power_10_100'] = 0.0
                    features['microphonics_q_factor'] = 0.0
            except:
                features['microphonics_dom_freq'] = 0.0
                features['microphonics_band_power_10_100'] = 0.0
                features['microphonics_q_factor'] = 0.0
        else:
            features['microphonics_rms'] = 0.0
            features['microphonics_dom_freq'] = 0.0
            features['microphonics_band_power_10_100'] = 0.0
            features['microphonics_q_factor'] = 0.0
    else:
        features['microphonics_rms'] = 0.0
        features['microphonics_dom_freq'] = 0.0
        features['microphonics_band_power_10_100'] = 0.0
        features['microphonics_q_factor'] = 0.0

    # ========================================================================
    # RF POWER FLOW AND BALANCE (5 features)
    # Signal definitions per thesis Table 2.2:
    #   - Uci: Incident/forward signal (computed from I/Q Uci)
    #   - A Ucr: Reflected signal amplitude (column 9)
    # Power is proportional to voltage squared: P ∝ V²
    # ========================================================================
    if 'Uci' in signals and 'A Ucr' in signals:
        uci = signals['Uci']
        ucr = signals['A Ucr']  # CORRECTED: Use A Ucr for reflected, not courant pickup

        trigger_idx = zero_idx  # V4: real trigger (was hardcoded 3000)

        # Forward and reflected power (proportional to voltage squared)
        forward_power = np.abs(uci)**2
        reflected_power = np.abs(ucr)**2  # CORRECTED: A Ucr is reflected signal amplitude

        # Pre-trigger means
        features['forward_power_mean'] = np.mean(forward_power[:trigger_idx]) if len(forward_power) > trigger_idx else np.mean(forward_power)
        features['reflected_power_mean'] = np.mean(reflected_power[:trigger_idx]) if len(reflected_power) > trigger_idx else np.mean(reflected_power)

        # Reflection coefficient (Prefl/Pfwd)
        ratio = features['reflected_power_mean'] / (features['forward_power_mean'] + 1e-10)
        features['power_reflection_ratio'] = ratio

        # Jump at trigger -- V4: window width scaled to a fixed physical
        # duration (was a fixed 10 samples, i.e. a physically meaningless
        # duration once NDEC varies)
        jump_half_window = scale_samples(10, dt_us)
        if len(forward_power) > trigger_idx + jump_half_window:
            pre_power = np.mean(forward_power[trigger_idx - jump_half_window:trigger_idx])
            post_power = np.mean(forward_power[trigger_idx:trigger_idx + jump_half_window])
            features['power_jump_at_trigger'] = np.abs(post_power - pre_power) / (pre_power + 1e-10)
        else:
            features['power_jump_at_trigger'] = 0.0

        # Power imbalance RMS (stability)
        if len(forward_power) > trigger_idx:
            power_diff = forward_power[:trigger_idx] - reflected_power[:trigger_idx]
            features['power_imbalance_rms'] = np.std(power_diff)
        else:
            features['power_imbalance_rms'] = 0.0
    else:
        features['forward_power_mean'] = 0.0
        features['reflected_power_mean'] = 0.0
        features['power_reflection_ratio'] = 0.0
        features['power_jump_at_trigger'] = 0.0
        features['power_imbalance_rms'] = 0.0

    # ========================================================================
    # 3. VALIDITY AND GATING FLAGS (5 flags)
    # ========================================================================

    # RF drive state (from forward power). V6 fix: use the unscaled signal --
    # the threshold (1.0) is calibrated against real Uci amplitude (typical
    # RF-on ~1.5-1.7, RF-off ~0.000001, passes_quality_filters()'s own
    # comment), which the Z-score normalization used to destroy, making this
    # flag constantly 0 regardless of the real RF-drive state.
    if 'Uci' in signals:
        forward_mean = np.mean(np.abs(signals_unscaled['Uci']))
        features['rf_drive_on'] = int(forward_mean > 1.0)  # Threshold to tune
    else:
        features['rf_drive_on'] = 0

    # LLRF loop state (from metadata)
    loop_status = metadata.get('LOOP', 'UNKNOWN')
    if isinstance(loop_status, str):
        features['llrf_loop_closed'] = int(loop_status.upper() in ['ON', 'OUI', 'CLOSED'])
    else:
        features['llrf_loop_closed'] = int(loop_status == 1)

    # Feed-forward enabled/disabled (from metadata). V7 RENAME (2026-09-09):
    # this reads the header field named "BEAM" -- despite the name, it does
    # NOT encode particle-beam presence, it encodes feed-forward state in the
    # LLRF control loop. Same field/logic as V6's 'beam_present', renamed only
    # -- see module docstring.
    #
    # CORRECTION (2026-09-09, same day): the BEAM field was not actually
    # populated before 2021 -- the recorded value (always "NON") is a stale
    # default, not a real measurement of that era's feed-forward state.
    # Per the project lead: assume feed-forward was ON by default pre-2021
    # (standard operating procedure predating this toggle's introduction),
    # regardless of what the unpopulated field literally says. Mirrors
    # Classify_PostMortemFile's own _get_acquisition_year() year-gating for
    # the filter fix -- same field, same "not meaningful before 2021" logic,
    # now applied to the feature value as well as the filter.
    acquisition_year = None
    date_str = metadata.get('DATE', '') or ''
    if isinstance(date_str, str):
        year_match = re.search(r'(19|20)\d{2}', date_str)
        if year_match:
            acquisition_year = int(year_match.group(0))

    if acquisition_year is not None and acquisition_year < 2021:
        features['feed_forward_enabled'] = 1
    else:
        feed_forward_status = metadata.get('BEAM', 'UNKNOWN')
        if isinstance(feed_forward_status, str):
            features['feed_forward_enabled'] = int(feed_forward_status.upper() in ['OUI', 'YES', 'ON'])
        else:
            features['feed_forward_enabled'] = int(feed_forward_status == 1)

    # V6: interlock_type (ALM-bit-derived) REMOVED -- it is a near-tautological
    # copy of the label (y_binary = alm > 0 uses the exact same ALM field this
    # feature was derived from), confirmed as the root cause of XGBoost's
    # literal 100%-on-every-metric result on V2. See module docstring.

    # Valid decay window (check if post-trigger has monotonic decay)
    # V4: real trigger position, and a physical-duration-scaled window
    # (was a fixed 100 samples).
    if 'Ucav' in signals:
        trigger_idx = zero_idx
        decay_check_window = scale_samples(100, dt_us)
        if len(signals['Ucav']) > trigger_idx + decay_check_window:
            decay_window = np.abs(signals['Ucav'][trigger_idx:trigger_idx + decay_check_window])
            # Check if mostly decreasing
            decreases = np.sum(np.diff(decay_window) < 0)
            features['valid_decay_window'] = int(decreases > 0.6 * len(decay_window))  # >60% samples decrease
        else:
            features['valid_decay_window'] = 0
    else:
        features['valid_decay_window'] = 0

    # ========================================================================
    # 4. PRECURSOR FEATURES (50+ features)
    # ========================================================================
    # V4: this is a bounded precursor window like preprocess_signals' segments
    # -- the delta_pre_samples worth of time IMMEDIATELY BEFORE the true
    # trigger (zero_idx), not "the first N samples of the file" (V2/V3's bug:
    # for most files that was hundreds of ms before the trigger, nowhere near
    # the actual lead-up to the fault this section is meant to characterize).
    # Same physical duration as preprocess_signals' segments, for consistency.
    delta_pre = scale_samples(3000, dt_us)
    early_len = scale_samples(1000, dt_us)
    late_start = scale_samples(2000, dt_us)

    for signal_name, signal_data in signals.items():
        window_start = max(0, zero_idx - delta_pre)
        pre_trigger = signal_data[window_start:zero_idx]
        if len(pre_trigger) < delta_pre:
            continue

        # Temporal trends (linear fit)
        x = np.arange(len(pre_trigger))
        coeffs = np.polyfit(x, pre_trigger, 1)
        features[f'{signal_name}_trend_slope'] = coeffs[0]

        # Quadratic trend (acceleration)
        coeffs_quad = np.polyfit(x, pre_trigger, 2)
        features[f'{signal_name}_trend_acceleration'] = coeffs_quad[0]

        # CUSUM (cumulative sum for anomaly detection)
        cusum = np.cumsum(pre_trigger - np.mean(pre_trigger))
        features[f'{signal_name}_cusum_max'] = np.max(cusum)
        features[f'{signal_name}_cusum_min'] = np.min(cusum)

        # Multi-window comparison (early vs late)
        early_window = pre_trigger[:early_len]
        late_window = pre_trigger[late_start:delta_pre]
        features[f'{signal_name}_early_late_diff'] = np.mean(late_window) - np.mean(early_window)
        # V6 FIX: the previous `!= 0` guard only caught an exactly-zero denominator,
        # not a near-zero one -- a handful of events have an early-window mean a few
        # orders of magnitude smaller than the signal's typical scale, producing
        # ratios of |z| > 20 relative to the rest of their fault category and
        # dominating downstream per-category clustering (found investigating
        # prepare_09c_enhanced_subclass_discovery.py's k=2 result, which was
        # isolating these single events rather than finding real subtype structure).
        # Match the same epsilon-relative-to-scale pattern already used by
        # var_ratio_late_early two lines below, instead of a bare != 0 check.
        _denom = np.mean(early_window)
        _eps = 1e-6 * (np.abs(pre_trigger).mean() + 1e-12)
        features[f'{signal_name}_early_late_ratio'] = (
            np.mean(late_window) / _denom if np.abs(_denom) > _eps else 0
        )

        # Variance ratio (early vs late) - precursor indicator
        var_early = np.var(early_window)
        var_late = np.var(late_window)
        features[f'{signal_name}_var_ratio_late_early'] = var_late / (var_early + 1e-10)

    # ========================================================================
    # 5. DERIVATIVE FEATURES (velocity, acceleration)
    # ========================================================================
    for signal_name in signals.keys():
        if signal_name in first_deriv:
            # 1st derivative (velocity)
            vel = first_deriv[signal_name]
            features[f'{signal_name}_velocity_mean'] = np.mean(vel)
            features[f'{signal_name}_velocity_std'] = np.std(vel)
            features[f'{signal_name}_velocity_max'] = np.max(np.abs(vel))

        if signal_name in second_deriv:
            # 2nd derivative (acceleration)
            accel = second_deriv[signal_name]
            features[f'{signal_name}_accel_mean'] = np.mean(accel)
            features[f'{signal_name}_accel_std'] = np.std(accel)
            features[f'{signal_name}_accel_max'] = np.max(np.abs(accel))

    # ========================================================================
    # 6. SEGMENT-BASED FEATURES (5 segments per signal)
    # ========================================================================
    for signal_name, signal_segments in segments.items():
        for segment_name, segment_data in signal_segments.items():
            if len(segment_data) > 0:
                features[f'{signal_name}_segment_{segment_name}_mean'] = np.mean(segment_data)
                features[f'{signal_name}_segment_{segment_name}_std'] = np.std(segment_data)
                # Trend within segment
                x_seg = np.arange(len(segment_data))
                if len(x_seg) > 1:
                    coeffs_seg = np.polyfit(x_seg, segment_data, 1)
                    features[f'{signal_name}_segment_{segment_name}_slope'] = coeffs_seg[0]
                else:
                    features[f'{signal_name}_segment_{segment_name}_slope'] = 0.0

    # ========================================================================
    # 7. METADATA FEATURES (~10)
    # ========================================================================
    for key, value in metadata.items():
        if isinstance(value, (int, float)):
            features[f'meta_{key}'] = value
        elif isinstance(value, str):
            try:
                features[f'meta_{key}'] = float(value)
            except ValueError:
                pass

    return features


def calculate_effective_decay(ucav_signal, zero_idx, fs=88000, decay_window_samples=500, nominal_tau=0.005):
    """
    Calculate effective system decay metrics (NOT Q_L)

    IMPORTANT: This does NOT calculate the intrinsic loaded quality factor Q_L.
    In CW operation with closed-loop LLRF control, the observed decay reflects:
    - RF drive removal dynamics
    - LLRF loop disengagement
    - Protection logic response
    - Cavity + RF chain dynamics

    The effective decay constant τ_eff encodes the system response, not intrinsic cavity loss.

    Args:
        ucav_signal: Cavity voltage time series (normalized)
        zero_idx: this file's true trigger sample index (V4 -- see module docstring)
        fs: Sampling frequency (Hz), per-file
        decay_window_samples: Number of samples to use for decay fit -- caller
            should pass scale_samples(500, dt_us) so this is a fixed physical
            duration (~5.68ms) regardless of NDEC, not a fixed sample count
        nominal_tau: Nominal decay time constant for reference (s), default 5ms

    Returns:
        (effective_decay_tau, decay_rate, decay_fit_r2, decay_relative_to_nominal, decay_non_exponentiality):
            effective_decay_tau: Fitted decay constant (s)
            decay_rate: 1/tau (Hz)
            decay_fit_r2: Goodness of fit
            decay_relative_to_nominal: Ratio to reference decay
            decay_non_exponentiality: RMS log-fit residual (loop interaction indicator)
    """
    # V4: trigger_idx is the real informatic trigger (was hardcoded 3000)
    trigger_idx = zero_idx
    if len(ucav_signal) < trigger_idx + decay_window_samples:
        return np.nan, np.nan, 0.0, np.nan, np.nan

    decay_signal = ucav_signal[trigger_idx:trigger_idx + decay_window_samples]
    decay_signal = np.abs(decay_signal)

    # Ensure decay (not growth)
    if decay_signal[0] < decay_signal[-1]:
        return np.nan, np.nan, 0.0, np.nan, np.nan

    # Remove offset and avoid log(0)
    decay_signal = decay_signal - decay_signal.min() + 1e-10

    t = np.arange(len(decay_signal)) / fs

    try:
        # Fit exponential decay: V(t) = V0 * exp(-t/tau)
        # Linearized: ln(V) = ln(V0) - t/tau
        log_signal = np.log(decay_signal)
        coeffs = np.polyfit(t, log_signal, 1)
        tau_eff = -1.0 / coeffs[0]

        # Decay rate (inverse of tau)
        decay_rate = 1.0 / tau_eff if tau_eff > 0 else np.nan

        # Calculate R² for fit quality
        y_pred = np.polyval(coeffs, t)
        ss_res = np.sum((log_signal - y_pred)**2)
        ss_tot = np.sum((log_signal - log_signal.mean())**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Relative to nominal (indicates faster/slower than typical)
        decay_relative_to_nominal = tau_eff / nominal_tau if nominal_tau > 0 else np.nan

        # Non-exponentiality: RMS of log-fit residuals
        # High values indicate non-exponential decay (e.g., active loop dynamics)
        residuals = log_signal - y_pred
        decay_non_exponentiality = np.sqrt(np.mean(residuals**2))

        return tau_eff, decay_rate, r2, decay_relative_to_nominal, decay_non_exponentiality

    except Exception:
        return np.nan, np.nan, 0.0, np.nan, np.nan


class LLRFDataPreparation:
    """LLRF Data Preparation Pipeline for Cluster Processing (Memory-Optimized V2)"""

    def __init__(self, data_dir, output_dir, n_jobs=1, max_files=None, resume=False,
                 batch_size=500, file_timeout=30, store_signals=False):
        """
        Initialize LLRF Data Preparation pipeline.

        Args:
            data_dir: Path to raw LLRF data files
            output_dir: Path to output directory for processed data
            n_jobs: Number of parallel workers (default: 1, recommended: 4)
            max_files: Maximum files to process (None = all)
            resume: Resume from last successful batch
            batch_size: Files per batch (default: 500, reduced from 2000 for memory)
            file_timeout: Timeout per file in seconds (default: 30)
            store_signals: Store full signals in batches (default: False, saves memory)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs
        self.max_files = max_files
        self.resume = resume
        self.batch_size = batch_size
        self.file_timeout = file_timeout
        self.store_signals = store_signals

        # Checkpoint files
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Number of jobs: {self.n_jobs}")
        logger.info(f"Max files: {self.max_files if self.max_files else 'All'}")
        logger.info(f"Resume: {self.resume}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"File timeout: {self.file_timeout}s")
        logger.info(f"Store signals: {self.store_signals} {'(HIGH MEMORY)' if self.store_signals else '(memory efficient)'}")

    def collect_file_paths(self):
        """Collect all LLRF data file paths recursively"""
        logger.info("Scanning for data files recursively...")

        # Check for checkpoint
        checkpoint_file = self.checkpoint_dir / 'file_paths.pkl'
        if self.resume and checkpoint_file.exists():
            logger.info(f"Loading file paths from checkpoint: {checkpoint_file}")
            with open(checkpoint_file, 'rb') as f:
                file_paths = pickle.load(f)
            logger.info(f"Loaded {len(file_paths)} file paths from checkpoint")
            return file_paths

        # Recursively scan directory for all files
        logger.info(f"Recursively searching in: {self.data_dir}")
        all_files = list(self.data_dir.glob('**/*'))

        # Filter files
        excluded_suffixes = ['.txt', '.md', '.json', '.pyc', '.log', '.xml', '.html',
                           '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.csv', '.fig']
        excluded_names = ['.directory', '.DS_Store', 'Thumbs.db']

        file_paths = []
        skip_stats = {
            'not_file': 0,
            'hidden': 0,
            'excluded_name': 0,
            'excluded_suffix': 0,
            'too_small': 0,
            'permission_denied': 0,
            'other_error': 0
        }

        for f in all_files:
            if not f.is_file():
                skip_stats['not_file'] += 1
                continue

            if f.name.startswith('.'):
                skip_stats['hidden'] += 1
                continue

            if f.name in excluded_names:
                skip_stats['excluded_name'] += 1
                continue

            if f.suffix.lower() in excluded_suffixes:
                skip_stats['excluded_suffix'] += 1
                continue

            try:
                file_stat = f.stat()
                if file_stat.st_size < 100:
                    skip_stats['too_small'] += 1
                    continue

                with open(f, 'rb') as test_f:
                    test_f.read(1)

            except PermissionError:
                skip_stats['permission_denied'] += 1
                continue
            except Exception:
                skip_stats['other_error'] += 1
                continue

            file_paths.append(f)

            if self.max_files and len(file_paths) >= self.max_files:
                break

        # Sort by path for consistency
        file_paths = sorted(file_paths)

        if self.max_files:
            file_paths = file_paths[:self.max_files]

        # Report statistics
        logger.info(f"Found {len(file_paths)} readable data files out of {len(all_files)} total items")
        logger.info("File filtering statistics:")
        for key, val in skip_stats.items():
            logger.info(f"  - {key}: {val}")

        if file_paths:
            logger.info(f"Sample files found:")
            for f in file_paths[:5]:
                logger.info(f"  - {f.relative_to(self.data_dir)} ({f.stat().st_size} bytes)")

        # Save checkpoint
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(file_paths, f)
        logger.info(f"Saved file paths checkpoint: {checkpoint_file}")

        return file_paths

    def run_parallel(self, file_paths):
        """Run data preparation with multiprocessing and batching (MEMORY-OPTIMIZED)"""
        import gc  # Import garbage collector

        logger.info(f"Running parallel processing with {self.n_jobs} jobs...")
        logger.info(f"Processing {len(file_paths)} files in batches of {self.batch_size}")
        logger.info(f"Memory mode: {'FULL SIGNALS' if self.store_signals else 'LIGHTWEIGHT (features only)'}")

        # Create batch directory
        batch_dir = self.output_dir / 'batches'
        batch_dir.mkdir(exist_ok=True)

        # Check for existing batches if resuming
        start_batch_idx = 0
        if self.resume:
            existing_batches = sorted(batch_dir.glob('batch_*.pkl'))
            if existing_batches:
                last_batch = existing_batches[-1]
                start_batch_idx = int(last_batch.stem.split('_')[1]) + 1  # +1 to start AFTER last batch
                logger.info(f"Resuming from batch {start_batch_idx} (found {len(existing_batches)} existing batches)")

        # Process in batches
        num_batches = (len(file_paths) + self.batch_size - 1) // self.batch_size
        all_batch_files = []

        # Track problematic files
        timeout_files = []
        error_files = []

        # Memory tracking
        process = psutil.Process()

        for batch_idx in range(start_batch_idx, num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(file_paths))
            batch_files = file_paths[start_idx:end_idx]

            mem_gb = process.memory_info().rss / 1e9
            logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_files)} files)... [Memory: {mem_gb:.2f} GB]")

            # Create partial function with timeout and store_signals parameters
            process_func = partial(
                process_single_file_with_timeout,
                file_timeout=self.file_timeout,
                store_signals=self.store_signals
            )

            with Pool(processes=self.n_jobs) as pool:
                batch_results = list(tqdm(
                    pool.imap(process_func, batch_files, chunksize=10),  # Added chunksize for efficiency
                    total=len(batch_files),
                    desc=f"Batch {batch_idx + 1}/{num_batches}"
                ))

            # Track failures
            for result in batch_results:
                if result and not result.get('success', False):
                    if 'TIMEOUT' in result.get('error', ''):
                        timeout_files.append(result['file'])
                    else:
                        error_files.append((result['file'], result.get('error', 'Unknown')))

            # Filter successful results
            batch_results = [r for r in batch_results if r and r.get('success', False)]

            # Save batch immediately
            batch_file = batch_dir / f'batch_{batch_idx:03d}.pkl'
            with open(batch_file, 'wb') as f:
                pickle.dump({
                    'results': batch_results,
                    'n_events': len(batch_results),
                    'batch_idx': batch_idx,
                    'store_signals': self.store_signals
                }, f)

            logger.info(f"Saved batch {batch_idx + 1}: {len(batch_results)} events → {batch_file}")
            all_batch_files.append(batch_file)

            # CRITICAL: Free memory aggressively
            del batch_results
            gc.collect()  # Force garbage collection after each batch

        # Save problematic files list
        if timeout_files:
            timeout_file = self.output_dir / 'timeout_files.txt'
            with open(timeout_file, 'w') as f:
                f.write(f"Files that timed out (>{self.file_timeout}s)\n")
                f.write("=" * 80 + "\n\n")
                for file in timeout_files:
                    f.write(f"{file}\n")
            logger.warning(f"⚠️  {len(timeout_files)} files timed out. List saved to: {timeout_file}")

        if error_files:
            error_file = self.output_dir / 'error_files.txt'
            with open(error_file, 'w') as f:
                f.write(f"Files that failed with errors\n")
                f.write("=" * 80 + "\n\n")
                for file, error in error_files[:100]:  # Limit to first 100
                    f.write(f"{file}: {error}\n")
            logger.warning(f"⚠️  {len(error_files)} files had errors. List saved to: {error_file}")

        # Don't load all batches into memory - return batch directory for incremental processing
        logger.info("Batches saved. Will process incrementally to avoid OOM...")
        logger.info(f"Found {len(list(batch_dir.glob('batch_*.pkl')))} batches ready for processing")
        return batch_dir  # Return directory, not merged results

    def save_results(self, batch_dir_or_results):
        """Save processed results to disk (MEMORY-EFFICIENT VERSION)

        Processes batches one at a time to avoid loading all data into memory.
        Creates outputs incrementally compatible with downstream pipeline.

        Args:
            batch_dir_or_results: Either Path to batch directory (new) or list of results (legacy)
        """
        import gc  # Import garbage collector

        # Handle both batch directory and results list
        if isinstance(batch_dir_or_results, Path):
            batch_dir = batch_dir_or_results
            logger.info("="*80)
            logger.info("INCREMENTAL PROCESSING MODE (Memory-Efficient)")
            logger.info("="*80)

            # =================================================================
            # PASS 1: Collect lightweight data (metadata, features, labels)
            # =================================================================
            logger.info("PASS 1: Collecting metadata, features, and labels...")

            metadata_list = []
            features_list = []
            fault_labels = []
            file_paths_list = []  # Track file paths for potential re-processing
            signal_names = None
            n_events_total = 0
            has_signals = None  # Track if batches contain signals

            batch_files = sorted(batch_dir.glob('batch_*.pkl'))
            for batch_idx, batch_file in enumerate(batch_files):
                logger.info(f"  [{batch_idx+1}/{len(batch_files)}] Processing {batch_file.name}...")

                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)
                    batch_results = batch_data['results']

                    if len(batch_results) == 0:
                        continue

                    # Check if this batch has signals stored
                    if has_signals is None:
                        has_signals = 'signals' in batch_results[0]
                        logger.info(f"    Batch mode: {'FULL SIGNALS' if has_signals else 'LIGHTWEIGHT (features only)'}")

                    # Get signal names from first result
                    if signal_names is None:
                        if has_signals:
                            signal_names = list(batch_results[0]['signals']['signals'].keys())
                        elif 'signal_names' in batch_results[0]:
                            signal_names = batch_results[0]['signal_names']
                        else:
                            signal_names = []  # Will be populated later if needed

                    # Extract lightweight data only
                    for result in batch_results:
                        metadata_list.append(result['metadata'])
                        features_list.append(result['features'])
                        fault_labels.append(result['fault_labels'])
                        file_paths_list.append(result['file'])
                        n_events_total += 1

                    del batch_data, batch_results
                    gc.collect()  # Force GC after each batch

            if len(metadata_list) == 0:
                logger.error("No results to save - all files failed to process")
                raise ValueError("No data was successfully processed")

            logger.info(f"✓ Collected {n_events_total} events from {len(batch_files)} batches")

        else:
            # Legacy: handle list of results
            results = batch_dir_or_results
            logger.info(f"Saving {len(results)} processed events...")

            if len(results) == 0:
                logger.error("No results to save - all files failed to process")
                raise ValueError("No data was successfully processed")

            # Separate components
            metadata_list = [r['metadata'] for r in results]
            features_list = [r['features'] for r in results]
            fault_labels = [r['fault_labels'] for r in results]
            signal_names = list(results[0]['signals']['signals'].keys()) if results else []
            n_events_total = len(results)

        # =================================================================
        # Create feature dataframe and labels from collected data
        # =================================================================
        logger.info("")
        logger.info("Creating feature matrices and labels...")
        df_features = pd.DataFrame(features_list)

        # Free features_list immediately (now stored in df_features)
        del features_list
        import gc
        gc.collect()
        logger.info(f"  Freed features_list from memory")

        # Add control loop alias features (for backward compatibility with notebooks)
        # These are aliases for existing features with the names expected by analysis notebooks
        logger.info("Adding control loop alias features...")
        alias_mappings = {
            'amp_rms_pre': 'rf_mismatch_rms_pre',
            'amp_rms_post': 'rf_mismatch_rms_post',
            'phase_rms_pre': 'phase_std_pre',
            'phase_rms_post': 'phase_std_post',
            'control_overshoot_amp': 'rf_mismatch_max_excursion',
            'control_settling_time': 'rf_mismatch_settling_time',
            'control_saturation_fraction': 'reflected_saturation_fraction',
        }
        for alias_name, source_name in alias_mappings.items():
            if source_name in df_features.columns and alias_name not in df_features.columns:
                df_features[alias_name] = df_features[source_name]
                logger.info(f"  Added alias: {alias_name} <- {source_name}")
        logger.info(f"  Total features after aliases: {len(df_features.columns)}")

        # Create labels
        y_binary = np.array([f['binary'] for f in fault_labels])
        y_multilabel = np.array([f['multilabel'] for f in fault_labels])
        y_defaut = np.array([f['defaut'] for f in fault_labels])
        y_physics_subtype = np.array([f['physics_subtype'] for f in fault_labels], dtype=object)
        y_alignment_refined = np.array([f.get('alignment_refined', False) for f in fault_labels])

        # Free fault_labels immediately (now stored in y_binary/y_multilabel/y_defaut/y_physics_subtype/y_alignment_refined)
        del fault_labels
        gc.collect()
        logger.info(f"  Freed fault_labels from memory")

        # Log fault vs no-fault distribution
        n_normal = np.sum(y_binary == 0)
        n_fault = np.sum(y_binary == 1)
        logger.info(f"")
        logger.info(f"="*80)
        logger.info(f"DATASET COMPOSITION (Critical for ML Training):")
        logger.info(f"="*80)
        logger.info(f"  Total events:      {len(y_binary)}")
        logger.info(f"  Normal (ground truth):  {n_normal} ({100*n_normal/len(y_binary):.1f}%)")
        logger.info(f"  Fault (ground truth):   {n_fault} ({100*n_fault/len(y_binary):.1f}%)")
        logger.info(f"  Balance ratio:   {n_fault}/{n_normal} = {n_fault/(n_normal+1e-10):.2f}:1")
        logger.info(f"")
        if n_normal == 0:
            logger.warning(f"⚠️  WARNING: NO NORMAL EVENTS DETECTED!")
            logger.warning(f"  ML training requires BOTH fault and no-fault events!")
            logger.warning(f"  Check if manual acquisitions are in the dataset")
        if n_fault == 0:
            logger.warning(f"⚠️  WARNING: NO FAULT EVENTS DETECTED!")
        logger.info(f"="*80)

        # =================================================================
        # PASS 2: Save lightweight preprocessed_data.pkl (skip signal accumulation)
        # =================================================================
        if isinstance(batch_dir_or_results, Path):
            logger.info("")
            logger.info("PASS 2: Saving lightweight preprocessed_data.pkl (skipping signal accumulation to avoid OOM)...")

            # Save reference-only file - sequences will be built directly in Pass 3
            preprocessed_file = self.output_dir / 'preprocessed_data.pkl'
            with open(preprocessed_file, 'wb') as f:
                pickle.dump({
                    'signals_normalized': 'BUILT_IN_PASS3',  # Placeholder
                    'metadata': metadata_list,
                    'signal_names': signal_names,
                    '_batch_directory': str(batch_dir),  # Reference to raw batches
                    'note': 'Full signal data available in features_engineered.pkl sequence arrays. Use features_engineered.pkl for ML workflows.',
                }, f)
            logger.info(f"✓ Saved lightweight preprocessed data: {preprocessed_file}")
            logger.info(f"  Skipped signal accumulation to avoid OOM")
            logger.info(f"  Sequence arrays will be built directly from batches in Pass 3")

        else:
            # Legacy mode: save directly
            signals_list = [r['signals'] for r in results]
            preprocessed_file = self.output_dir / 'preprocessed_data.pkl'
            with open(preprocessed_file, 'wb') as f:
                pickle.dump({
                    'signals_normalized': signals_list,
                    'metadata': metadata_list,
                    'signal_names': signal_names,
                }, f)
            logger.info(f"Saved preprocessed data: {preprocessed_file}")

        # Save features
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # Step 1: Correlation filtering - remove highly correlated features
        # IMPORTANT: Physics features are PROTECTED from filtering
        logger.info("Applying feature selection (correlation filtering)...")
        feature_cols = df_features.columns.tolist()
        X_features = df_features.fillna(0).values

        # Define physics features to protect from correlation filtering
        # These are domain-specific features that should always be kept
        PROTECTED_PHYSICS_FEATURES = {
            # Effective Decay (5 features)
            'effective_decay_tau', 'decay_rate', 'decay_fit_r2',
            'decay_relative_to_nominal', 'decay_non_exponentiality',
            # Detuning (5 features)
            'detuning_mean_hz', 'detuning_std_hz', 'detuning_slope_hz_s',
            'detuning_peak_to_peak', 'detuning_jump_at_trigger',
            # Microphonics (4 features)
            'microphonics_rms', 'microphonics_dom_freq',
            'microphonics_band_power_10_100', 'microphonics_q_factor',
            # Phase Stability (4 features)
            'phase_jitter_rms', 'phase_excursion_pp',
            'phase_noise_slope', 'phase_psd_integral',
            # RF Power Flow (5 features)
            'forward_power_mean', 'reflected_power_mean', 'power_reflection_ratio',
            'power_jump_at_trigger', 'power_imbalance_rms',
            # Control Loop (7 features): NOT protected -- these are pure Python-level
            # aliases of the "RF Mismatch" features below (features['amp_rms_pre'] =
            # features.get('rf_mismatch_rms_pre', 0.0), etc., see the "CONTROL LOOP
            # FEATURES - ALIASES" block above), so they correlate at EXACTLY 1.0 with
            # their source and should be deduplicated by the normal >0.95 correlation
            # filter like any other redundant feature -- protecting them here defeated
            # that, silently doubling their weight in every downstream model and
            # diluting feature-importance rankings (found 2026-09-05: control_overshoot_amp
            # and rf_mismatch_max_excursion were showing up as two separate features with
            # identical values). Still computed (for notebook name-compatibility via
            # features_all), just no longer exempted from dedup.
            # RF Mismatch (original names, also protected)
            'rf_mismatch_rms_pre', 'rf_mismatch_rms_post', 'rf_mismatch_max_excursion',
            'rf_mismatch_settling_time', 'reflected_saturation_fraction',
            'phase_std_pre', 'phase_std_post',
            # Validity Flags (5 features)
            'rf_drive_on', 'llrf_loop_closed', 'feed_forward_enabled',
            'valid_decay_window',
            # Modulator features
            'mod_amplitude_mean_pre', 'mod_amplitude_std_pre', 'mod_phase_std_pre',
            'mod_amplitude_change', 'mod_phase_change', 'mod_amplitude_max_post',
        }

        corr_matrix = np.corrcoef(X_features.T)
        threshold = 0.95
        to_drop = set()

        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > threshold:
                    # Only drop if the feature is NOT protected
                    if feature_cols[j] not in PROTECTED_PHYSICS_FEATURES:
                        to_drop.add(feature_cols[j])

        # Count how many physics features were protected
        protected_count = sum(1 for f in feature_cols if f in PROTECTED_PHYSICS_FEATURES)
        logger.info(f"Protected physics features: {protected_count}")
        logger.info(f"Features to drop (corr > {threshold}): {len(to_drop)}/{len(feature_cols)}")

        feature_cols_filtered = [col for col in feature_cols if col not in to_drop]
        df_features_filtered = df_features[feature_cols_filtered]
        X_filtered = df_features_filtered.fillna(0).values

        logger.info(f"Remaining features after correlation filtering: {len(feature_cols_filtered)}")

        # Clean data: replace inf/nan with finite values
        n_inf = np.isinf(X_filtered).sum()
        n_nan = np.isnan(X_filtered).sum()
        if n_inf > 0 or n_nan > 0:
            logger.warning(f"Found {n_inf} inf values and {n_nan} nan values in features. Replacing with finite values...")
            X_filtered = np.nan_to_num(X_filtered, nan=0.0, posinf=1e10, neginf=-1e10)
            logger.info(f"✓ Cleaned feature matrix: all values now finite")

        # Step 2: Standardization (zero mean, unit variance)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)

        # Step 3: PCA dimensionality reduction (95% variance retention)
        n_samples, n_features = X_scaled.shape
        if n_samples > 1 and n_features > 1:
            pca = PCA(n_components=0.95, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            logger.info(f"PCA results:")
            logger.info(f"  Original features: {X_scaled.shape[1]}")
            logger.info(f"  PCA components: {X_pca.shape[1]}")
            logger.info(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")
        else:
            logger.warning(f"Not enough samples/features for PCA ({n_samples} samples, {n_features} features)")
            pca = None
            X_pca = X_scaled

        # =================================================================
        # PASS 3: Build sequence arrays (only if signals are stored)
        # =================================================================
        if isinstance(batch_dir_or_results, Path):
            if has_signals:
                logger.info("")
                logger.info("PASS 3: Building sequence arrays incrementally (MEMORY-MAPPED MODE)...")

                n_events = n_events_total
                n_samples = 4000  # Standard LLRF acquisition length
                n_signals = len(signal_names)

                # Pre-allocate arrays using memory-mapped files (avoids RAM limits)
                logger.info(f"  Pre-allocating memory-mapped sequence arrays for {n_events} events...")
                logger.info(f"  Using temporary memory-mapped files to avoid RAM limits")

                # Create memory-mapped arrays in the output directory
                mmap_dir = self.output_dir / '_temp_mmap'
                mmap_dir.mkdir(exist_ok=True)

                sequences_full = np.memmap(
                    str(mmap_dir / 'sequences_full.dat'),
                    dtype=np.float32, mode='w+',
                    shape=(n_events, n_samples, n_signals)
                )
                sequences_pretrigger = np.memmap(
                    str(mmap_dir / 'sequences_pretrigger.dat'),
                    dtype=np.float32, mode='w+',
                    shape=(n_events, 3000, n_signals)
                )
                sequences_downsampled = np.memmap(
                    str(mmap_dir / 'sequences_downsampled.dat'),
                    dtype=np.float32, mode='w+',
                    shape=(n_events, n_samples // 10, n_signals)
                )

                logger.info(f"  Memory-mapped files created in: {mmap_dir}")

                # Fill arrays batch-by-batch
                event_offset = 0
                for batch_idx, batch_file in enumerate(batch_files):
                    logger.info(f"  [{batch_idx+1}/{len(batch_files)}] Processing sequences from {batch_file.name}...")

                    with open(batch_file, 'rb') as f:
                        batch_data = pickle.load(f)
                        batch_results = batch_data['results']

                        if len(batch_results) == 0:
                            continue

                        # Process each event in this batch
                        for local_idx, result in enumerate(batch_results):
                            global_idx = event_offset + local_idx
                            signals_dict = result['signals']['signals']

                            for j, signal_name in enumerate(signal_names):
                                if signal_name in signals_dict:
                                    sig = signals_dict[signal_name]
                                    sig_len = min(len(sig), n_samples)

                                    # Full sequence (zero-padded if shorter than 4000)
                                    sequences_full[global_idx, :sig_len, j] = sig[:sig_len]

                                    # Pre-trigger portion (first 3000 samples)
                                    pretrig_len = min(len(sig), 3000)
                                    sequences_pretrigger[global_idx, :pretrig_len, j] = sig[:pretrig_len]

                                    # Downsampled (every 10th sample)
                                    downsampled = sig[::10]
                                    downsamp_len = min(len(downsampled), n_samples // 10)
                                    sequences_downsampled[global_idx, :downsamp_len, j] = downsampled[:downsamp_len]

                        event_offset += len(batch_results)
                        del batch_data, batch_results
                        gc.collect()  # Force garbage collection after each batch

                logger.info(f"✓ Sequence arrays built successfully (memory-mapped)")

                # Flush memory-mapped arrays to disk
                logger.info("  Flushing memory-mapped arrays to disk...")
                sequences_full.flush()
                sequences_pretrigger.flush()
                sequences_downsampled.flush()

                # Convert to regular numpy arrays for pickling
                # Note: This loads data back into RAM, but does it efficiently
                logger.info("  Converting memory-mapped arrays to numpy arrays for saving...")
                sequences_full_np = np.array(sequences_full, dtype=np.float32)
                sequences_pretrigger_np = np.array(sequences_pretrigger, dtype=np.float32)
                sequences_downsampled_np = np.array(sequences_downsampled, dtype=np.float32)

                # Delete memory-mapped file references
                del sequences_full, sequences_pretrigger, sequences_downsampled
                gc.collect()

                # Reassign for use below
                sequences_full = sequences_full_np
                sequences_pretrigger = sequences_pretrigger_np
                sequences_downsampled = sequences_downsampled_np

                logger.info("  ✓ Arrays converted successfully")

                # Clean up temporary memory-mapped files
                import shutil
                try:
                    shutil.rmtree(mmap_dir)
                    logger.info(f"  Cleaned up temporary memory-mapped files: {mmap_dir}")
                except Exception as e:
                    logger.warning(f"  Could not remove temporary directory {mmap_dir}: {e}")
            else:
                # LIGHTWEIGHT MODE: No signals stored, create empty placeholder arrays
                logger.info("")
                logger.info("PASS 3: SKIPPED - Lightweight mode (no signals stored)")
                logger.info("  Sequence arrays will be empty placeholders.")
                logger.info("  To build sequence arrays, re-run with --store-signals flag.")

                n_events = n_events_total
                n_samples = 4000
                n_signals = len(signal_names) if signal_names else 27  # Default 27 signals

                # Create minimal placeholder arrays (just metadata shape info)
                sequences_full = np.zeros((n_events, 1, 1), dtype=np.float32)
                sequences_pretrigger = np.zeros((n_events, 1, 1), dtype=np.float32)
                sequences_downsampled = np.zeros((n_events, 1, 1), dtype=np.float32)

                # Save file paths for potential future re-processing
                file_paths_file = self.output_dir / 'processed_file_paths.pkl'
                with open(file_paths_file, 'wb') as f:
                    pickle.dump(file_paths_list, f)
                logger.info(f"  Saved file paths for future sequence building: {file_paths_file}")

        else:
            # Legacy mode: build from signals_list
            logger.info("Reconstructing sequence arrays for temporal/sequence-based models...")

            n_events = len(metadata_list)
            n_samples = 4000
            n_signals = len(signal_names)

            sequences_full = np.zeros((n_events, n_samples, n_signals), dtype=np.float32)
            sequences_pretrigger = np.zeros((n_events, 3000, n_signals), dtype=np.float32)
            sequences_downsampled = np.zeros((n_events, n_samples // 10, n_signals), dtype=np.float32)

            signals_list = [r['signals'] for r in results]
            for i, signal_data in enumerate(signals_list):
                signals_dict = signal_data['signals']

                for j, signal_name in enumerate(signal_names):
                    if signal_name in signals_dict:
                        sig = signals_dict[signal_name]
                        sig_len = min(len(sig), n_samples)

                        sequences_full[i, :sig_len, j] = sig[:sig_len]
                        pretrig_len = min(len(sig), 3000)
                        sequences_pretrigger[i, :pretrig_len, j] = sig[:pretrig_len]
                        downsampled = sig[::10]
                        downsamp_len = min(len(downsampled), n_samples // 10)
                        sequences_downsampled[i, :downsamp_len, j] = downsampled[:downsamp_len]

            del signals_list

        logger.info(f"Sequence arrays reconstructed:")
        logger.info(f"  sequences_full: {sequences_full.shape}")
        logger.info(f"  sequences_pretrigger: {sequences_pretrigger.shape}")
        logger.info(f"  sequences_downsampled: {sequences_downsampled.shape}")

        # V7: distinct filename -- do NOT overwrite V2/V3/V4/V5/V6's outputs, all
        # real checkpoints. Callers switch to this explicitly once validated.
        features_file = self.output_dir / 'features_engineered_v7.pkl'
        with open(features_file, 'wb') as f:
            pickle.dump({
                # Feature matrices
                'features_all': df_features,
                'feature_cols': feature_cols_filtered,  # Filtered features (after correlation removal)
                'X_scaled': X_scaled,
                'X_pca': X_pca,

                # Labels
                'y_binary': y_binary,
                'y_multilabel': y_multilabel,
                'fault_column_names': FAULT_CATEGORIES,
                'y_defaut': y_defaut,                    # per-event primary category (string), incl. 'Normal'
                'y_physics_subtype': y_physics_subtype,   # finer label for RF-regulation events; None otherwise
                'y_alignment_refined': y_alignment_refined,  # True where V5's AMPT onset refinement fired
                'label_source': str(GROUND_TRUTH_CSV),
                'alignment_note': 'V4: per-file trigger alignment from Read_Signals time axis (zero_idx/dt_us), not hardcoded trigger_idx=3000. V5: additionally refines onset via AMPT (find_true_t0_with_ampt) for "Rég signal RF hors tolérance" events only. V6: additionally drops the interlock_type feature (ALM-derived, was a near-tautological label proxy -- see module docstring). V7: ground truth re-run against the feed-forward-filter fix (recovers legitimate pre-2021 events); beam_present renamed feed_forward_enabled to match its real meaning. See module docstring.',

                # Transformers
                'scaler': scaler,
                'pca': pca,

                # Metadata
                'metadata': metadata_list,
                'signal_names': signal_names,

                # Sequence arrays for temporal/sequence models (Step 05 precursor detection)
                'sequences_full': sequences_full,
                'sequences_pretrigger': sequences_pretrigger,
                'sequences_downsampled': sequences_downsampled,
            }, f)
        logger.info(f"Saved features: {features_file}")
        logger.info(f"  Including {n_signals} signal channels × {n_events} events")
        logger.info(f"  Sequence arrays: full={sequences_full.shape}, pretrig={sequences_pretrigger.shape}, downsampled={sequences_downsampled.shape}")

        # Save summary (V7: distinct filename, don't overwrite V2/V3/V4/V5/V6's summaries)
        summary_file = self.output_dir / 'processing_summary_v7.txt'
        with open(summary_file, 'w') as f:
            f.write(f"LLRF Data Preparation Summary (V7 -- V6 + feed-forward-filter fix, recovers pre-2021 events)\n")
            f.write(f"Events with AMPT onset refinement applied: {int(np.sum(y_alignment_refined))}\n")
            f.write(f"=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Label source: {GROUND_TRUTH_CSV}\n")
            f.write(f"Total events processed: {len(metadata_list)}\n")
            f.write(f"Normal events: {np.sum(y_binary == 0)}\n")
            f.write(f"Fault events: {np.sum(y_binary == 1)}\n")
            f.write(f"Features extracted: {X_scaled.shape[1]}\n")
            f.write(f"PCA components: {X_pca.shape[1]}\n")
        logger.info(f"Saved summary: {summary_file}")

        logger.info("")
        logger.info("="*80)
        logger.info("ALL PROCESSING COMPLETE")
        logger.info("="*80)

        return {
            'n_events': len(metadata_list),
            'n_normal': np.sum(y_binary == 0),
            'n_fault': np.sum(y_binary == 1),
            'n_features': X_scaled.shape[1]
        }

    def run(self):
        """Main execution pipeline"""
        start_time = time.time()

        process = psutil.Process()
        logger.info(f"Initial memory usage: {process.memory_info().rss / 1e9:.2f} GB")

        # Step 1: Collect file paths
        file_paths = self.collect_file_paths()

        # Step 2: Process files
        if self.n_jobs > 1:
            batch_dir_or_results = self.run_parallel(file_paths)
        else:
            logger.error("Single-process mode not supported in this version. Use --n-jobs > 1")
            raise ValueError("Use --n-jobs > 1")

        # Check success based on return type
        if isinstance(batch_dir_or_results, Path):
            # Batch directory - event count will be determined during save_results
            n_batches = len(list(batch_dir_or_results.glob('batch_*.pkl')))
            logger.info(f"Successfully created {n_batches} batches from {len(file_paths)} input files")
            if n_batches == 0:
                logger.error("FATAL: No batches were successfully created!")
                raise RuntimeError("No data was successfully processed")
        else:
            # Legacy: results is a list
            logger.info(f"Successfully processed {len(batch_dir_or_results)}/{len(file_paths)} files")
            if len(batch_dir_or_results) == 0:
                logger.error("FATAL: No files were successfully processed!")
                raise RuntimeError("No data was successfully processed")

        # Step 3: Save results
        summary = self.save_results(batch_dir_or_results)

        # Final stats
        elapsed = time.time() - start_time
        logger.info(f"")
        logger.info(f"="*80)
        logger.info(f"FINAL STATISTICS")
        logger.info(f"="*80)
        logger.info(f"Memory usage: {process.memory_info().rss / 1e9:.2f} GB")
        logger.info(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        logger.info(f"Processing rate: {summary['n_events']/elapsed:.2f} files/second")
        logger.info(f"")
        logger.info(f"Summary:")
        logger.info(f"  Events: {summary['n_events']}")
        logger.info(f"  Normal: {summary['n_normal']}")
        logger.info(f"  Faults: {summary['n_fault']}")
        logger.info(f"  Features: {summary['n_features']}")
        logger.info(f"="*80)

        return summary


def main():
    parser = argparse.ArgumentParser(
        description='LLRF Data Preparation for Cluster (MEMORY-OPTIMIZED V2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to LLRF data directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--n-jobs', type=int, default=4,
                        help='Number of parallel jobs (default: 4, use -1 for all CPUs)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last successful batch')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Batch size for processing (default: 500, reduced for memory efficiency)')
    parser.add_argument('--file-timeout', type=int, default=30,
                        help='Timeout per file in seconds (default: 30s)')
    parser.add_argument('--store-signals', action='store_true',
                        help='Store full signal arrays in batch files (HIGH MEMORY - not recommended for large datasets)')

    args = parser.parse_args()

    # Determine number of jobs
    if args.n_jobs == -1:
        n_jobs = cpu_count()
        logger.info(f"Using all available CPUs: {n_jobs}")
    else:
        n_jobs = args.n_jobs

    # Run pipeline
    pipeline = LLRFDataPreparation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_jobs=n_jobs,
        max_files=args.max_files,
        resume=args.resume,
        batch_size=args.batch_size,
        file_timeout=args.file_timeout,
        store_signals=args.store_signals
    )

    try:
        summary = pipeline.run()
        logger.info("✓ Data preparation completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"✗ Data preparation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
