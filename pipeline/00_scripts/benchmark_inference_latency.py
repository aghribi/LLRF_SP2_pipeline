#!/usr/bin/env python3
"""
Real inference-latency benchmark, replacing the unverified "45ms/event" /
"<100ms" claims in the report (which predate this session's leakage fixes and
the V6 rebuild -- see PHASE3_WEAKNESS_DIAGNOSIS_AND_OPERATIONAL_USABILITY.md
section 2.5, "Latency"). No timing benchmark existed anywhere in this
pipeline before this script.

Times single-event, single-threaded inference (no batching -- the realistic
"one new postmortem file arrives" deployment scenario) for three already-
trained V6 models: step 06's binary RF, Phase C's Label Powerset (XGBoost,
this session's best multi-label approach per 07_multilabel_phaseABCD.yaml),
and step 05's corrected precursor RF. For each, times scaler.transform()
followed by model.predict() per row, one row at a time, over that model's own
test set, and reports mean/median/p95 wall-clock ms/event.
"""
import os
import sys
import time
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import __main__

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utilities.reporting.manifest import save_manifest

sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import raw_feature_matrix
import phaseC_lp_test
__main__.LabelPowersetWrapper = phaseC_lp_test.LabelPowersetWrapper  # for unpickling lp_model.pkl

# 2026-09-10: was hardcoded to cooked_data_v6, the same reproducibility trap
# already fixed in the SHAP scripts -- parameterized the same way
# (SPIRAL2_COOKED_DIR, default V6 for back-compat). VERSION_SUFFIX (e.g. "v6",
# "v7") feeds the two step-output subdirectory names below that carry an
# explicit version suffix; step_09_phaseC_lp does not and is used as-is.
COOKED_DIR = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6'))
VERSION_SUFFIX = COOKED_DIR.name.replace('cooked_data', '').lstrip('_') or 'v6'
N_TIMED = 200  # events timed per model (single-event calls; test sets are 313-633 events)


def time_single_event_inference(scaler, predict_fn, X_raw_rows, n_timed):
    """Time scaler.transform + predict_fn, one row at a time. Returns array of ms."""
    n = min(n_timed, X_raw_rows.shape[0])
    times_ms = np.empty(n)
    for i in range(n):
        row = X_raw_rows[i:i+1]
        t0 = time.perf_counter()
        row_scaled = scaler.transform(row)
        _ = predict_fn(row_scaled)
        t1 = time.perf_counter()
        times_ms[i] = (t1 - t0) * 1000.0
    return times_ms


def summarize(name, times_ms):
    print(f"\n{name}: n={len(times_ms)}")
    print(f"  mean={times_ms.mean():.3f}ms  median={np.median(times_ms):.3f}ms  "
          f"p95={np.percentile(times_ms, 95):.3f}ms  max={times_ms.max():.3f}ms")
    return {
        'mean_ms': float(times_ms.mean()),
        'median_ms': float(np.median(times_ms)),
        'p95_ms': float(np.percentile(times_ms, 95)),
        'max_ms': float(times_ms.max()),
        'n_timed': int(len(times_ms)),
    }


def main():
    with open(COOKED_DIR / 'features_engineered.pkl', 'rb') as f:
        data = pickle.load(f)

    summary = {}

    # --- Step 06: binary RF ---
    with open(COOKED_DIR / f'step_06_phase1_{VERSION_SUFFIX}' / 'binary_classification.pkl', 'rb') as f:
        d6 = pickle.load(f)
    splits6 = d6['data_splits']
    X_raw6, _ = raw_feature_matrix({'features_all': data['features_all'], 'feature_cols': splits6['feature_cols']})
    X_test_raw6 = X_raw6[splits6['idx_test']]
    rf6 = d6['random_forest']['model']
    times6 = time_single_event_inference(
        splits6['scaler'], lambda r: rf6.predict(r), X_test_raw6, N_TIMED
    )
    summary['step06_binary_rf'] = summarize('Step 06 binary RF', times6)

    # --- Phase C: Label Powerset (XGBoost) ---
    with open(COOKED_DIR / 'step_09_phaseC_lp' / 'lp_model.pkl', 'rb') as f:
        dc = pickle.load(f)
    X_rawC, _ = raw_feature_matrix({'features_all': data['features_all'], 'feature_cols': dc['feature_cols']})
    X_test_rawC = X_rawC[dc['idx_test']]
    lp_model = dc['model']
    timesC = time_single_event_inference(
        dc['scaler'], lambda r: lp_model.predict(r), X_test_rawC, N_TIMED
    )
    summary['phaseC_label_powerset'] = summarize('Phase C Label Powerset (XGBoost)', timesC)

    # --- Step 05: corrected precursor RF (382-column true-precursor subset) ---
    with open(COOKED_DIR / f'step_05_{VERSION_SUFFIX}' / 'precursor_detection.pkl', 'rb') as f:
        d5 = pickle.load(f)
    splits5 = d5['data_splits']
    X_raw5, _ = raw_feature_matrix({'features_all': data['features_all'], 'feature_cols': splits5['feature_cols']})
    X_test_raw5 = X_raw5[splits5['idx_test']]
    rf5 = d5['random_forest']['model']
    times5 = time_single_event_inference(
        splits5['scaler'], lambda r: rf5.predict_proba(r), X_test_raw5, N_TIMED
    )
    summary['step05_precursor_rf'] = summarize('Step 05 precursor RF (corrected, 382 features)', times5)

    metrics = {}
    for model_key, s in summary.items():
        metrics[f'{model_key}_mean_ms'] = {'value': s['mean_ms'], 'fmt': '.3f',
                                            'label': f'{model_key}: mean inference latency (ms/event)'}
        metrics[f'{model_key}_p95_ms'] = {'value': s['p95_ms'], 'fmt': '.3f',
                                           'label': f'{model_key}: p95 inference latency (ms/event)'}

    save_manifest(
        phase='inference_latency_benchmark',
        metrics=metrics,
        pipeline_run={
            'dataset_version': VERSION_SUFFIX.upper(),
            'dataset_path': str(COOKED_DIR / 'features_engineered.pkl'),
            'script': 'pipeline/00_scripts/benchmark_inference_latency.py',
        },
        meta={
            'method': (
                'Single-event, single-threaded wall-clock timing (scaler.transform + predict, one row '
                f'at a time, no batching) over each model\'s own {VERSION_SUFFIX.upper()} test set -- the realistic '
                '"one new postmortem file arrives" deployment scenario. Replaces the unverified '
                '"45ms/event"/"<100ms" claims, which predate this session\'s leakage fixes and V6, '
                'with a real, reproducible number. Feature engineering time (reading the raw postmortem '
                'file, computing the ~766/382 features) is NOT included -- this times model inference '
                'only, given per-file feature engineering already runs as part of the existing offline '
                'pipeline, not per-event online in this benchmark\'s scope.'
            ),
            'full_summary': summary,
        },
    )
    print("\nSaved manifest: analysis/results_manifest/inference_latency_benchmark.yaml")


if __name__ == '__main__':
    main()
