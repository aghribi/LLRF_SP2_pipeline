#!/usr/bin/env python3
"""Run Phase C SHAP analysis headless: create per-class summary and top-20 plots and save metrics CSV."""
import os
from pathlib import Path
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import shap
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('phaseC_shap_run')

COOKED = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')
PHASEC = COOKED / 'step_09_phaseC_lp'
SHAP_FILE = PHASEC / 'shap_lp.npz'
SUMMARY_FILE = PHASEC / 'phaseC_summary.pkl'
OUT_DIR = Path('analysis/outputs/phaseC_shap')
OUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info('SHAP file: %s', SHAP_FILE)
logger.info('Summary file: %s', SUMMARY_FILE)

if not SHAP_FILE.exists():
    logger.error('SHAP file not found: %s', SHAP_FILE)
    raise SystemExit(1)

# Load shap archive
sh = np.load(SHAP_FILE, allow_pickle=True)
if SUMMARY_FILE.exists():
    with open(SUMMARY_FILE, 'rb') as f:
        summary = pickle.load(f)
else:
    logger.warning('Summary file not found: %s — proceeding without summary', SUMMARY_FILE)
    summary = {}

label_names = list(sh['label_names']) if 'label_names' in sh else [f'F{i}' for i in range(len([k for k in sh.keys() if str(k).startswith('shap_')] ))]
feature_names = list(sh['feature_names']) if 'feature_names' in sh else None

# If SHAP archive doesn't include feature names, try to load them from the features artifact
if feature_names is None:
    feat_paths = [COOKED / 'step_03_features' / 'features_engineered.pkl', COOKED / 'features_engineered.pkl']
    for p in feat_paths:
        try:
            if p.exists():
                with open(p, 'rb') as f:
                    feat = pickle.load(f)
                fn = feat.get('feature_names') or feat.get('feature_cols')
                if fn is not None:
                    feature_names = list(fn)
                    logger.info('Loaded feature_names from %s', p)
                    break
        except Exception as e:
            logger.warning('Failed to load feature names from %s: %s', p, e)

logger.info('Labels: %s', label_names)
logger.info('Feature names available: %s', feature_names is not None)

# For each label/class, generate plots
for i, lbl in enumerate(label_names):
    key = f'shap_{i}'
    if key not in sh:
        logger.warning('No SHAP for label %s -> %s', lbl, key)
        continue
    shap_vals = sh[key]
    try:
        plt.figure(figsize=(8,6))
        try:
            shap.summary_plot(shap_vals, feature_names=feature_names, show=False)
            plt.tight_layout()
            outp = OUT_DIR / f'shap_summary_label_{i}.png'
            plt.savefig(outp, bbox_inches='tight')
            plt.clf()
            logger.info('Saved shap summary for %s -> %s', lbl, outp)
        except Exception as e:
            logger.warning('shap.summary_plot failed for %s: %s', lbl, e)
            plt.clf()

        # Prepare a 2D SHAP array (n_samples, n_features) for aggregation
        if isinstance(shap_vals, list):
            if len(shap_vals) >= 2:
                shap_2d = np.asarray(shap_vals[-1])
            else:
                shap_2d = np.asarray(shap_vals[0])
        else:
            arr = np.asarray(shap_vals)
            if arr.ndim == 2:
                shap_2d = arr
            elif arr.ndim >= 3:
                if arr.shape[-1] == 2:
                    shap_2d = arr[..., 1]
                else:
                    shap_2d = np.mean(arr, axis=tuple(range(2, arr.ndim)))
            else:
                shap_2d = arr.reshape((arr.shape[0], -1))

        mean_abs = np.mean(np.abs(shap_2d), axis=0)
        mean_abs = np.ravel(mean_abs)
        idx = np.argsort(mean_abs)[-20:]
        fig, ax = plt.subplots(figsize=(8,6))
        ax.barh(range(len(idx)), mean_abs[idx])
        ax.set_yticks(range(len(idx)))
        names = [(feature_names[j] if feature_names is not None else f'feat_{j}') for j in idx]
        ax.set_yticklabels(names)
        ax.set_title(f'Top 20 features by mean |SHAP| (class {i} - {lbl})')
        outp2 = OUT_DIR / f'top20_label_{i}.png'
        fig.savefig(outp2, bbox_inches='tight')
        plt.close(fig)
        logger.info('Saved top20 for %s -> %s', lbl, outp2)
    except Exception as e:
        logger.exception('Failed to produce plots for label %s: %s', lbl, e)

# Metrics (LP)
lp = summary.get('lp')
rows = []
if lp:
    rows.append({'model':'LP','hamming':lp.get('hamming_loss'),'micro_f1':lp.get('micro_f1'),'macro_f1':lp.get('macro_f1')})
if rows:
    df = pd.DataFrame(rows).set_index('model')
    csv_path = OUT_DIR / 'phaseC_metrics.csv'
    df.to_csv(csv_path)
    logger.info('Saved metrics to %s', csv_path)
    print(df)
else:
    logger.warning('No LP metrics found in summary')

logger.info('Done. Outputs in %s', OUT_DIR)
