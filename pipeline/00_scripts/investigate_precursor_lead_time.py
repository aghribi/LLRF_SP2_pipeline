#!/usr/bin/env python3
"""
Precursor lead-time measurement.

Background: the report claims a precursor "lead time" ("50-500ms before fault
onset" in one place, "50-400ms" in another) with no computational basis
anywhere in this pipeline -- step 05's corrected model (see
PHASE3_WEAKNESS_DIAGNOSIS_AND_OPERATIONAL_USABILITY.md section 2.3,
leakage_safe_features.true_precursor_columns()) scores a single FIXED window
immediately before each event's own trigger index (zero_idx); it never asks
how much *earlier* than that the signal remains distinguishable.

Method: reuse prepare_data_cluster_v6.py's own preprocess_signals()/
engineer_features() functions unchanged, but call them with a "probe" zero_idx
shifted progressively earlier than the file's real zero_idx -- this guarantees
the recomputed features are built by the exact same code path the model was
trained on, just anchored at a different point in the signal, rather than a
reimplementation that could subtly diverge. For a sample of true-positive
fault test events (correctly flagged as fault by the RF at offset 0), probe
a grid of offsets before zero_idx, score each with step 05's already-trained
RF + already-fit scaler, and find the largest offset at which predicted fault
probability still exceeds 0.5 -- i.e. how far back detectability holds.
Converts each file's own offset (in samples) to milliseconds via its own
dt_us, and reports the distribution across the sample.
"""
import sys
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, '/pbs/throng/m4cast/projects/SPIRAL2/programmes/PyPostMortem/src')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utilities.reporting.manifest import save_manifest

sys.path.insert(0, str(Path(__file__).parent))
from prepare_data_cluster_v6 import preprocess_signals, engineer_features, scale_samples

V6_DIR = '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6'
N_SAMPLE = 40
# Offsets tested, in units of the training window's own physical duration
# (delta_pre = scale_samples(3000, dt_us), ~34ms at the dominant NDEC=200).
# A first pass with multiples up to 8x found EVERY sampled event still >=0.5
# probability at the largest offset tested (identical 272.6ms for all 40,
# std=0.0 -- the tell that the grid ceiling, not a genuine detectability
# boundary, was being measured). Widened substantially here; probe_offset()
# returns None once a file's own buffer runs out, so this is now a
# right-censored measurement: some events will exhaust available buffer
# before probability ever drops below 0.5, and that must be reported as
# censored, not silently folded into the same distribution as events with a
# genuine observed drop.
OFFSET_MULTIPLES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]


def get_signals_and_zero_idx(path):
    from PyPostmortem.utils.PyPostMortem import Read_Signals
    with open(path, 'rb') as f:
        content = f.read()
    parameters, time_arr, df_signals, _, _, _ = Read_Signals(
        content, compute_defauts=False, compute_etats=False, plot_signaux=False, show_header=False
    )
    metadata = {}
    for key, val_dict in parameters.items():
        if isinstance(val_dict, dict) and 'Valeur' in val_dict:
            metadata[key] = val_dict['Valeur']
        else:
            metadata[key] = val_dict
    signals = {col: df_signals[col].values for col in df_signals.columns}
    time_arr = np.asarray(time_arr)
    zero_idx = int(np.argmin(np.abs(time_arr)))
    dt_us = float(np.median(np.diff(time_arr))) if len(time_arr) > 1 else 1.0
    return signals, metadata, zero_idx, dt_us


def probe_offset(signals, metadata, probe_zero_idx, dt_us, feature_cols):
    """Recompute the trained feature set with zero_idx shifted to probe_zero_idx."""
    if probe_zero_idx <= 0:
        return None
    processed = preprocess_signals(signals, metadata, probe_zero_idx, dt_us)
    features = engineer_features(processed, metadata, probe_zero_idx, dt_us)
    row = {c: features.get(c, np.nan) for c in feature_cols}
    x = np.array([row[c] for c in feature_cols], dtype=float)
    if np.any(np.isnan(x)):
        return None  # window too short at this offset for this file (guarded features absent)
    return x


def main():
    with open(f'{V6_DIR}/step_05_v6/precursor_detection.pkl', 'rb') as f:
        precursor = pickle.load(f)
    with open(f'{V6_DIR}/features_engineered_v6.pkl', 'rb') as f:
        data = pickle.load(f)
    with open(f'{V6_DIR}/processed_file_paths.pkl', 'rb') as f:
        paths = pickle.load(f)

    rf_model = precursor['random_forest']['model']
    scaler = precursor['data_splits']['scaler']
    feature_cols = precursor['data_splits']['feature_cols']
    idx_test = precursor['data_splits']['idx_test']
    y_test = precursor['data_splits']['y_test']
    y_pred_test = precursor['random_forest']['predictions']

    # True positives only: correctly-flagged fault events at offset 0 -- these
    # are the events for which "how much earlier is it still detectable" is a
    # meaningful question.
    true_positive_test_idx = idx_test[(y_test == 1) & (y_pred_test == 1)]
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(true_positive_test_idx,
                             size=min(N_SAMPLE, len(true_positive_test_idx)), replace=False)
    print(f"True positive fault events in test set: {len(true_positive_test_idx)}; "
          f"sampling {len(sample_idx)}")

    results = []       # genuine drop below 0.5 observed within available buffer
    censored = []       # ran out of buffer while still >=0.5 -- true lead time is >= this
    for i in sample_idx:
        path = paths[i]
        try:
            signals, metadata, zero_idx_real, dt_us = get_signals_and_zero_idx(path)
        except Exception as e:
            print(f"  FAILED to read {path}: {e}")
            continue

        delta_pre = scale_samples(3000, dt_us)
        max_detected_offset_samples = None
        genuine_drop_observed = False
        probed = {}
        for mult in OFFSET_MULTIPLES:
            offset_samples = int(round(mult * delta_pre))
            probe_zero_idx = zero_idx_real - offset_samples
            x = probe_offset(signals, metadata, probe_zero_idx, dt_us, feature_cols)
            if x is None:
                break  # buffer exhausted at this offset -- censored regardless of what came before
            x_scaled = scaler.transform(x.reshape(1, -1))
            proba = rf_model.predict_proba(x_scaled)[0, 1]
            probed[mult] = float(proba)
            if proba >= 0.5:
                max_detected_offset_samples = offset_samples
            else:
                genuine_drop_observed = True
                break  # genuine drop below threshold -- this offset is the boundary

        if max_detected_offset_samples is None:
            continue  # never detected even at offset 0 for this "true positive" -- shouldn't happen, skip defensively
        lead_time_ms = max_detected_offset_samples * dt_us / 1000.0
        # Censored unless we actually WATCHED probability cross below 0.5 --
        # running out of buffer OR running out of OFFSET_MULTIPLES while still
        # >=0.5 are both censoring, not a measured boundary.
        if genuine_drop_observed:
            results.append({'lead_time_ms': lead_time_ms, 'probed_probas': probed})
        else:
            censored.append({'lead_time_ms_lower_bound': lead_time_ms, 'probed_probas': probed})

    n_evaluated = len(results)
    n_censored = len(censored)
    print(f"\nCensored (ran out of available buffer before probability ever dropped below 0.5): "
          f"{n_censored} of {n_evaluated + n_censored}")
    if censored:
        censored_lb = np.array([c['lead_time_ms_lower_bound'] for c in censored])
        print(f"  Their lower-bound lead times (true value is >= this): "
              f"mean={censored_lb.mean():.1f}ms median={np.median(censored_lb):.1f}ms "
              f"max={censored_lb.max():.1f}ms")
    n_total = n_evaluated + n_censored
    print(f"\n{n_evaluated} of {n_total} events showed a genuine drop below 0.5 within available buffer; "
          f"{n_censored} of {n_total} were censored (buffer ran out first).")

    metrics = {
        'n_events_total': {'value': n_total, 'fmt': None, 'label': 'True-positive fault events evaluated'},
        'n_censored': {'value': n_censored, 'fmt': None,
                        'label': 'Events where buffer ran out before probability ever dropped below 0.5'},
        'censored_fraction': {'value': (n_censored / n_total) if n_total else 0.0, 'fmt': '.1%',
                               'label': 'Fraction of evaluated events that were censored'},
    }
    meta_extra = {}

    if results:
        lead_times = np.array([r['lead_time_ms'] for r in results])
        print(f"Genuine (uncensored) lead times (ms before interlock trip, boundary actually observed):")
        print(f"  mean={lead_times.mean():.1f}  median={np.median(lead_times):.1f}  "
              f"min={lead_times.min():.1f}  max={lead_times.max():.1f}  std={lead_times.std():.1f}")
        metrics.update({
            'lead_time_ms_mean_uncensored': {'value': float(lead_times.mean()), 'fmt': '.1f',
                                              'label': 'Mean lead time, genuine drop observed (ms)'},
            'lead_time_ms_median_uncensored': {'value': float(np.median(lead_times)), 'fmt': '.1f',
                                                'label': 'Median lead time, genuine drop observed (ms)'},
        })
    else:
        print("No event showed a genuine drop below 0.5 within the tested/available buffer -- "
              "every evaluated event is censored (see below).")

    if censored:
        censored_lb = np.array([c['lead_time_ms_lower_bound'] for c in censored])
        print(f"Censored events' lower-bound lead times (true value is >= this):")
        print(f"  mean={censored_lb.mean():.1f}  median={np.median(censored_lb):.1f}  "
              f"max={censored_lb.max():.1f}")
        metrics.update({
            'lead_time_ms_lower_bound_mean_censored': {'value': float(censored_lb.mean()), 'fmt': '.1f',
                                                         'label': 'Mean lower-bound lead time for censored events (ms)'},
            'lead_time_ms_lower_bound_max_censored': {'value': float(censored_lb.max()), 'fmt': '.1f',
                                                        'label': 'Max lower-bound lead time for censored events (ms)'},
        })

    save_manifest(
        phase='05_precursor_lead_time',
        metrics=metrics,
        pipeline_run={
            'dataset_version': 'V6',
            'dataset_path': f'{V6_DIR}/features_engineered_v6.pkl',
            'script': 'pipeline/00_scripts/investigate_precursor_lead_time.py',
        },
        meta={
            'method': (
                'For each sampled true-positive fault event, reused prepare_data_cluster_v6.py\'s '
                'own preprocess_signals()/engineer_features() unchanged, calling them with zero_idx '
                'shifted progressively earlier than the file\'s real trigger index (offsets in units '
                f'of the training window\'s own physical duration: {OFFSET_MULTIPLES}), scored each '
                'with step 05\'s already-trained RF + already-fit scaler, and recorded the largest '
                'offset at which predicted probability still exceeded 0.5. Events where the file\'s own '
                'buffer ran out before probability ever dropped are CENSORED (true lead time is >= the '
                'last successfully-probed offset, not equal to it) and reported separately, not pooled '
                'with genuine boundary observations.'
            ),
            'offset_multiples_tested': OFFSET_MULTIPLES,
            'headline_finding': (
                f"{n_censored} of {n_total} sampled true-positive events were censored: detection "
                "confidence never dropped below 0.5 within the available pre-trigger buffer (up to "
                "several hundred ms tested). This means the report cannot cite a specific bounded lead "
                "time from this measurement -- the honest statement is 'detectable at least "
                f"~{(np.array([c['lead_time_ms_lower_bound'] for c in censored]).mean() if censored else 0):.0f}ms "
                "before the interlock trip for most sampled events, true extent not established', not a "
                "precise duration like the report's fabricated '50-500ms'. This also raises a further "
                "question worth flagging rather than resolving here: confidence this stable across "
                "hundreds of ms may indicate the model is partly keying on a stable per-capture "
                "characteristic rather than a fault-onset-proximate signal specifically -- see "
                "PHASE3_WEAKNESS_DIAGNOSIS_AND_OPERATIONAL_USABILITY.md section 2.3 for the related "
                "acquisition-convention caveat found earlier."
            ),
            'caveat': (
                'This measures how far before the INFORMATIC trigger (zero_idx) the signal remains '
                'distinguishable to this model, not how far before the TRUE PHYSICAL onset -- '
                'zero_idx itself can lag true onset (see the AMPT-refinement work in '
                'prepare_data_cluster_v6.py, applied to one of the seven fault categories). Report '
                'this as "detectable ahead of the interlock trip," not "before any physical change."'
            ),
        },
    )
    print("\nSaved manifest: analysis/results_manifest/05_precursor_lead_time.yaml")


if __name__ == '__main__':
    main()
