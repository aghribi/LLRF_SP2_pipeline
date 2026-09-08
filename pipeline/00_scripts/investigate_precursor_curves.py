#!/usr/bin/env python3
"""
Capture the FULL probability-vs-offset curve (not just the crossing point) for a
sample of true-positive fault events, reusing step 05's already-trained RF model
exactly as investigate_precursor_lead_time.py does. Purpose: directly check
whether detection confidence ramps up approaching the trigger (genuine precursor
signal) or stays flat across hundreds of ms (the "stable per-capture characteristic"
concern flagged in PHASE3_WEAKNESS_DIAGNOSIS...md section 2.3 / the lead-time
manifest's headline_finding) -- this session's own follow-up exploration, not yet
a results-manifest metric.
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
OUT_PATH = Path(V6_DIR) / 'step_05_v6' / 'precursor_curves_exploration.json'
N_SAMPLE = 20
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
    if probe_zero_idx <= 0:
        return None
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

    true_positive_test_idx = idx_test[(y_test == 1) & (y_pred_test == 1)]
    rng = np.random.RandomState(1)  # different seed from the original investigation, independent sample
    sample_idx = rng.choice(true_positive_test_idx, size=min(N_SAMPLE, len(true_positive_test_idx)), replace=False)
    print(f"True positive fault events in test set: {len(true_positive_test_idx)}; sampling {len(sample_idx)}", flush=True)

    curves = []
    for i in sample_idx:
        path = paths[i]
        try:
            signals, metadata, zero_idx_real, dt_us = get_signals_and_zero_idx(path)
        except Exception as e:
            print(f"  FAILED {path}: {e}", flush=True)
            continue
        delta_pre = scale_samples(3000, dt_us)
        probed_ms = {}
        for mult in OFFSET_MULTIPLES:
            offset_samples = int(round(mult * delta_pre))
            probe_zero_idx = zero_idx_real - offset_samples
            x = probe_offset(signals, metadata, probe_zero_idx, dt_us, feature_cols)
            if x is None:
                break
            x_scaled = scaler.transform(x.reshape(1, -1))
            proba = rf_model.predict_proba(x_scaled)[0, 1]
            offset_ms = offset_samples * dt_us / 1000.0
            probed_ms[offset_ms] = float(proba)
        curves.append({'idx': int(i), 'path': str(path), 'probed_ms': probed_ms})
        print(f"  idx={i}: {len(probed_ms)} offsets probed, probs={[round(v,3) for v in probed_ms.values()]}", flush=True)

    import json
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(curves, f, indent=2)
    print(f"\nSaved {len(curves)} curves -> {OUT_PATH}", flush=True)


if __name__ == '__main__':
    main()
