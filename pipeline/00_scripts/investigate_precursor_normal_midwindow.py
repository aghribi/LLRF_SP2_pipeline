#!/usr/bin/env python3
"""
Disambiguating test named in report/sections/06_results.tex and 07_discussion.tex: score the
same 386-column true-precursor feature set on a window drawn from the MIDDLE of a Normal event's
own recording, far from any acquisition boundary and unrelated to that file's real ALM trigger
(zero_idx). If the trained RF still confidently predicts 'fault' on these windows, that is direct
evidence the model keys on a stable per-capture/acquisition characteristic rather than genuine
fault-onset proximity (the concern raised by investigate_precursor_curves.py's flat probability
curves and the 97.5%-censored lead-time result). If it confidently predicts 'normal', that
specific concern is weakened.
"""
import sys
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np

sys.path.insert(0, '/pbs/throng/m4cast/projects/SPIRAL2/programmes/PyPostMortem/src')
sys.path.insert(0, '/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration')
sys.path.insert(0, '/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration/pipeline/00_scripts')
from prepare_data_cluster_v6 import preprocess_signals, engineer_features, scale_samples

V6_DIR = '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6'
OUT_PATH = Path(V6_DIR) / 'step_05_v6' / 'precursor_normal_midwindow_exploration.json'
N_SAMPLE = 20
BOUNDARY_MARGIN_FACTOR = 2.0  # require this many window-widths of clearance from either file edge


def get_signals_and_time(path):
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
    dt_us = float(np.median(np.diff(time_arr))) if len(time_arr) > 1 else 1.0
    return signals, metadata, time_arr, dt_us


def probe(signals, metadata, probe_zero_idx, dt_us, feature_cols):
    processed = preprocess_signals(signals, metadata, probe_zero_idx, dt_us)
    features = engineer_features(processed, metadata, probe_zero_idx, dt_us)
    row = {c: features.get(c, np.nan) for c in feature_cols}
    x = np.array([row[c] for c in feature_cols], dtype=float)
    if np.any(np.isnan(x)):
        return None
    return x


def main():
    with open(f'{V6_DIR}/step_05_v6/precursor_detection.pkl', 'rb') as f:
        precursor = pickle.load(f)
    with open(f'{V6_DIR}/processed_file_paths.pkl', 'rb') as f:
        paths = pickle.load(f)

    rf_model = precursor['random_forest']['model']
    scaler = precursor['data_splits']['scaler']
    feature_cols = precursor['data_splits']['feature_cols']
    idx_test = precursor['data_splits']['idx_test']
    y_test = precursor['data_splits']['y_test']
    y_pred_test = precursor['random_forest']['predictions']

    true_negative_test_idx = idx_test[(y_test == 0) & (y_pred_test == 0)]
    rng = np.random.RandomState(3)
    sample_idx = rng.choice(true_negative_test_idx, size=min(N_SAMPLE, len(true_negative_test_idx)), replace=False)
    print(f'Confidently-normal events in test set: {len(true_negative_test_idx)}; sampling {len(sample_idx)}', flush=True)

    results = []
    for i in sample_idx:
        path = paths[i]
        try:
            signals, metadata, time_arr, dt_us = get_signals_and_time(path)
        except Exception as e:
            print(f'  FAILED {path}: {e}', flush=True)
            continue
        n_samples = len(time_arr)
        delta_pre = scale_samples(3000, dt_us)
        mid_idx = n_samples // 2
        margin = int(round(BOUNDARY_MARGIN_FACTOR * delta_pre))
        if mid_idx - margin <= 0 or mid_idx + margin >= n_samples:
            print(f'  SKIP idx={i}: file too short for a boundary-clear middle window (n_samples={n_samples}, need margin={margin})', flush=True)
            continue
        x = probe(signals, metadata, mid_idx, dt_us, feature_cols)
        if x is None:
            print(f'  SKIP idx={i}: NaN features at mid-window', flush=True)
            continue
        x_scaled = scaler.transform(x.reshape(1, -1))
        proba_fault = float(rf_model.predict_proba(x_scaled)[0, 1])
        results.append({
            'idx': int(i), 'path': str(path), 'n_samples': int(n_samples),
            'mid_idx': int(mid_idx), 'proba_fault_at_midwindow': proba_fault,
        })
        print(f'  idx={i}: n_samples={n_samples}, mid_idx={mid_idx}, proba_fault={proba_fault:.4f}', flush=True)

    import json
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    n = len(results)
    n_flagged = sum(1 for r in results if r['proba_fault_at_midwindow'] > 0.5)
    print(f'\nSaved {n} results -> {OUT_PATH}', flush=True)
    print(f'{n_flagged}/{n} middle-of-Normal-recording windows scored as fault-like (proba > 0.5)', flush=True)


if __name__ == '__main__':
    main()
