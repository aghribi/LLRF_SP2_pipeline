#!/usr/bin/env python3
"""Phase B: Classifier Chains smoke test and scaffold.

Creates a small ClassifierChain model (XGBoost if available, else RandomForest),
performs a quick train/test split, saves model and summary to the cooked data area.
"""
import logging
import sys
from pathlib import Path
import pickle
import numpy as np
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utilities.reporting.manifest import save_manifest

sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import leakage_safe_split, repeated_leakage_safe_eval, compute_scale_pos_weight, true_precursor_columns
from split_diagnostics import log_split_composition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('phaseB_cc')

import os
COOKED = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data'))
PRETRIGGER = os.environ.get('SPIRAL2_PRETRIGGER', '0') == '1'
PHASE_SUFFIX = '_pretrigger' if PRETRIGGER else ''
DATASET_VERSION = (lambda _n: _n.upper() if _n else 'V2')(COOKED.name.replace('cooked_data', '').lstrip('_'))
PHASEB = COOKED / f'step_08_phaseB_cc{PHASE_SUFFIX}'
PHASEB.mkdir(parents=True, exist_ok=True)

def load_features():
    candidates = [COOKED / 'step_03_features' / 'features_engineered.pkl', COOKED / 'features_engineered.pkl']
    for p in candidates:
        if p.exists():
            logger.info('Loading features from %s', p)
            with open(p, 'rb') as f:
                obj = pickle.load(f)
            y = obj.get('y_multilabel')
            if PRETRIGGER:
                # Same restriction as prepare_05_phase0_precursor.py / step 06b.
                obj = dict(obj)
                obj['feature_cols'] = true_precursor_columns(obj['feature_cols'])
            return obj, y, obj.get('feature_cols')
    logger.warning('No features artifact found; generating synthetic data for smoke test')
    # synthetic multilabel: 200 samples, 50 features, 6 labels
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 50))
    y = (rng.rand(200, 6) > 0.8).astype(int)
    feature_names = [f'feat_{i}' for i in range(X.shape[1])]
    return {'synthetic_X': X}, y, feature_names

def choose_base_estimator(y_train=None):
    # Phase 4 (2026-09-06): class-weighted by default when y_train is given --
    # scale_pos_weight (XGBoost, data-driven aggregate; see
    # compute_scale_pos_weight()'s docstring for the per-label limitation) or
    # class_weight='balanced' (RandomForest fallback).
    try:
        from xgboost import XGBClassifier
        logger.info('Using XGBClassifier as base estimator (low-iter smoke test)')
        spw = compute_scale_pos_weight(y_train) if y_train is not None else 1.0
        return XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric='logloss',
                              scale_pos_weight=spw, verbosity=0)
    except Exception:
        logger.info('xgboost unavailable, using RandomForestClassifier')
        return RandomForestClassifier(n_estimators=50, class_weight='balanced')

def main():
    pkl_data, y, feature_names = load_features()
    if y is None:
        logger.error('Features or labels not available')
        return

    if 'synthetic_X' in pkl_data:
        # Smoke-test fallback: no real leakage concern (data is fabricated).
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            pkl_data['synthetic_X'], y, test_size=0.25, random_state=0
        )
        split_extra = {}
    else:
        # Leakage-safe split: scaler fit on the TRAIN fold only. y is
        # multi-label (2D) so stratify is off, matching the original code.
        split = leakage_safe_split(pkl_data, y, test_size=0.25, random_state=0,
                                    stratify=False)
        X_train, X_test = split['X_train'], split['X_test']
        y_train, y_test = split['y_train'], split['y_test']
        label_names = pkl_data.get('fault_column_names') or [f'label_{i}' for i in range(y.shape[1])]
        log_split_composition(y_train, y_test, label_names, logger,
                               title='Phase B (Classifier Chain) split composition')
        split_extra = {'scaler': split['scaler'], 'pca': split['pca'],
                        'feature_cols': split['feature_cols'],
                        'idx_train': split['idx_train'], 'idx_test': split['idx_test']}

    base = choose_base_estimator(y_train)
    cc = ClassifierChain(base)
    logger.info('Fitting ClassifierChain (this may take a moment)')
    cc.fit(X_train, y_train)
    logger.info('Predicting')
    y_pred = cc.predict(X_test)
    # metrics
    ham = hamming_loss(y_test, y_pred)
    micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    summary = {'cc': {'hamming_loss': float(ham), 'micro_f1': float(micro), 'macro_f1': float(macro)}}
    # save model (+ the scaler/feature_cols it was trained with, so downstream
    # SHAP extraction transforms new rows the same way) and summary
    model_path = PHASEB / 'cc_model.pkl'
    summary_path = PHASEB / 'phaseB_summary.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': cc, **split_extra}, f)
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    logger.info('Saved model -> %s', model_path)
    logger.info('Saved summary -> %s', summary_path)
    logger.info('Metrics: hamming=%.6f micro_f1=%.6f macro_f1=%.6f', ham, micro, macro)

    if 'synthetic_X' not in pkl_data:
        # Phase 4 (2026-09-06): baseline (unweighted) vs. class-weighted, so
        # the effect is measured, not assumed.
        logger.info('Repeated-split stability check (10 splits, different random_state each): '
                    'baseline (unweighted) vs. class-weighted...')

        def model_fn_baseline(X_train, y_train):
            m = ClassifierChain(choose_base_estimator(None))
            m.fit(X_train, y_train)
            return m

        def model_fn_weighted(X_train, y_train):
            m = ClassifierChain(choose_base_estimator(y_train))
            m.fit(X_train, y_train)
            return m

        metric_fns = {
            'macro_f1': lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
            'per_label_f1': lambda yt, yp: f1_score(yt, yp, average=None, zero_division=0),
            'hamming_loss': lambda yt, yp: hamming_loss(yt, yp),
        }
        stability_baseline = repeated_leakage_safe_eval(
            pkl_data, y, model_fn_baseline, metric_fns, n_repeats=10,
            test_size=0.25, stratify=False, base_random_state=0, label_names=label_names,
        )
        stability = repeated_leakage_safe_eval(
            pkl_data, y, model_fn_weighted, metric_fns, n_repeats=10,
            test_size=0.25, stratify=False, base_random_state=0, label_names=label_names,
        )
        logger.info('  macro_f1: baseline mean=%.4f -> weighted mean=%.4f',
                    stability_baseline['macro_f1']['mean'], stability['macro_f1']['mean'])
        logger.info('  Per-label F1, baseline -> weighted:')
        for lname in label_names:
            b = stability_baseline['per_label_f1']['per_label'][lname]
            a = stability['per_label_f1']['per_label'][lname]
            logger.info('    [%s]: %.3f+/-%.3f -> %.3f+/-%.3f', lname, b['mean'], b['std'], a['mean'], a['std'])

        save_manifest(
            phase=f'08_phaseB_cc_stability{PHASE_SUFFIX}',
            metrics={
                'single_split_macro_f1': {'value': float(macro), 'fmt': '.4f', 'label': 'Single-split (seed=0) CC macro-F1'},
            },
            pipeline_run={
                'dataset_version': DATASET_VERSION,
                'dataset_path': str(COOKED / 'step_03_features' / 'features_engineered.pkl'),
                'script': 'pipeline/00_scripts/phaseB_cc_test.py',
            },
            stability=stability,
            meta={'baseline_unweighted_stability': stability_baseline, 'pretrigger': PRETRIGGER},
        )

if __name__ == '__main__':
    main()
