#!/usr/bin/env python3
"""Phase C: Label Powerset smoke test and scaffold.

Trains a LabelPowerset classifier (XGBoost if available else RandomForest) on available features
or synthetic data for a quick smoke test. Saves model and summary to cooked data.
"""
import logging
import sys
from pathlib import Path
import pickle
import numpy as np
try:
    from sklearn.multiclass import LabelPowerset
except Exception:
    LabelPowerset = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utilities.reporting.manifest import save_manifest

sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import leakage_safe_split, repeated_leakage_safe_eval, true_precursor_columns
from split_diagnostics import log_split_composition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('phaseC_lp')

import os
COOKED = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data'))
PRETRIGGER = os.environ.get('SPIRAL2_PRETRIGGER', '0') == '1'
PHASE_SUFFIX = '_pretrigger' if PRETRIGGER else ''
DATASET_VERSION = (lambda _n: _n.upper() if _n else 'V2')(COOKED.name.replace('cooked_data', '').lstrip('_'))
PHASEC = COOKED / f'step_09_phaseC_lp{PHASE_SUFFIX}'
PHASEC.mkdir(parents=True, exist_ok=True)


class LabelPowersetWrapper:
    """Simple Label Powerset wrapper that maps unique label-tuples to integer classes.

    This is a lightweight fallback used when sklearn's LabelPowerset is unavailable.
    """
    def __init__(self, estimator, class_weight=None):
        self.estimator = estimator
        self.mapping = {}
        self.inv_map = {}
        # Phase 4 (2026-09-06): LabelPowerset collapses the multi-label problem
        # into ONE multiclass problem over label-combinations, so per-label
        # class_weight/scale_pos_weight don't apply here -- sample_weight
        # computed from the powerset classes' own frequencies is the correct,
        # estimator-agnostic equivalent (works for both the XGBoost and
        # RandomForest choose_base_estimator() branches).
        self.class_weight = class_weight

    def fit(self, X, Y):
        tuples = [tuple(map(int, row)) for row in Y]
        uniq = {}
        curr = 0
        y_enc = []
        for t in tuples:
            if t not in uniq:
                uniq[t] = curr
                curr += 1
            y_enc.append(uniq[t])
        self.mapping = uniq
        self.inv_map = {v: k for k, v in uniq.items()}
        y_enc = np.array(y_enc)
        if self.class_weight == 'balanced':
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight('balanced', y_enc)
            self.estimator.fit(X, y_enc, sample_weight=sample_weight)
        else:
            self.estimator.fit(X, y_enc)

    def predict(self, X):
        y_enc = self.estimator.predict(X)
        out = []
        for v in y_enc:
            tup = self.inv_map.get(int(v))
            if tup is None:
                out.append([0] * len(next(iter(self.mapping.keys()))))
            else:
                out.append(list(tup))
        return np.array(out)


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
    rng = np.random.RandomState(1)
    X = rng.normal(size=(200, 50))
    y = (rng.rand(200, 6) > 0.8).astype(int)
    feature_names = [f'feat_{i}' for i in range(X.shape[1])]
    return {'synthetic_X': X}, y, feature_names

def choose_base_estimator():
    try:
        from xgboost import XGBClassifier
        logger.info('Using XGBClassifier as base estimator')
        return XGBClassifier(n_estimators=30, use_label_encoder=False, eval_metric='logloss')
    except Exception:
        logger.info('xgboost unavailable, using RandomForestClassifier')
        return RandomForestClassifier(n_estimators=50)

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
        label_names_lp = pkl_data.get('fault_column_names') or [f'label_{i}' for i in range(y.shape[1])]
        log_split_composition(y_train, y_test, label_names_lp, logger,
                               title='Phase C (Label Powerset) split composition')
        split_extra = {'scaler': split['scaler'], 'pca': split['pca'],
                        'feature_cols': split['feature_cols'],
                        'idx_train': split['idx_train'], 'idx_test': split['idx_test']}

    base = choose_base_estimator()
    if LabelPowerset is not None:
        lp = LabelPowerset(base)
    else:
        # Use the module-level LabelPowersetWrapper fallback (picklable).
        # Phase 4 (2026-09-06): class_weight='balanced' -- sample_weight
        # computed from the powerset classes' own frequencies, see
        # LabelPowersetWrapper.fit().
        lp = LabelPowersetWrapper(base, class_weight='balanced')
    logger.info('Fitting LabelPowerset (this may take a moment)')
    lp.fit(X_train, y_train)
    logger.info('Predicting')
    y_pred = lp.predict(X_test)
    ham = hamming_loss(y_test, y_pred)
    micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    summary = {'lp': {'hamming_loss': float(ham), 'micro_f1': float(micro), 'macro_f1': float(macro)}}
    model_path = PHASEC / 'lp_model.pkl'
    summary_path = PHASEC / 'phaseC_summary.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': lp, **split_extra}, f)
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    logger.info('Saved model -> %s', model_path)
    logger.info('Saved summary -> %s', summary_path)
    logger.info('Metrics: hamming=%.6f micro_f1=%.6f macro_f1=%.6f', ham, micro, macro)

    if 'synthetic_X' not in pkl_data:
        # Phase 4 (2026-09-06): baseline (unweighted) vs. sample-weighted
        # ('balanced' over the powerset classes), so the effect is measured.
        logger.info('Repeated-split stability check (10 splits, different random_state each): '
                    'baseline (unweighted) vs. class-weighted...')

        def model_fn_baseline(X_train, y_train):
            base = choose_base_estimator()
            m = LabelPowerset(base) if LabelPowerset is not None else LabelPowersetWrapper(base)
            m.fit(X_train, y_train)
            return m

        def model_fn_weighted(X_train, y_train):
            base = choose_base_estimator()
            m = LabelPowerset(base) if LabelPowerset is not None else LabelPowersetWrapper(base, class_weight='balanced')
            m.fit(X_train, y_train)
            return m

        metric_fns = {
            'macro_f1': lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
            'per_label_f1': lambda yt, yp: f1_score(yt, yp, average=None, zero_division=0),
            'hamming_loss': lambda yt, yp: hamming_loss(yt, yp),
        }
        stability_baseline = repeated_leakage_safe_eval(
            pkl_data, y, model_fn_baseline, metric_fns, n_repeats=10,
            test_size=0.25, stratify=False, base_random_state=0, label_names=label_names_lp,
        )
        stability = repeated_leakage_safe_eval(
            pkl_data, y, model_fn_weighted, metric_fns, n_repeats=10,
            test_size=0.25, stratify=False, base_random_state=0, label_names=label_names_lp,
        )
        logger.info('  macro_f1: baseline mean=%.4f -> weighted mean=%.4f',
                    stability_baseline['macro_f1']['mean'], stability['macro_f1']['mean'])
        logger.info('  Per-label F1, baseline -> weighted:')
        for lname in label_names_lp:
            b = stability_baseline['per_label_f1']['per_label'][lname]
            a = stability['per_label_f1']['per_label'][lname]
            logger.info('    [%s]: %.3f+/-%.3f -> %.3f+/-%.3f', lname, b['mean'], b['std'], a['mean'], a['std'])

        save_manifest(
            phase=f'09_phaseC_lp_stability{PHASE_SUFFIX}',
            metrics={
                'single_split_macro_f1': {'value': float(macro), 'fmt': '.4f', 'label': 'Single-split (seed=0) LP macro-F1'},
            },
            pipeline_run={
                'dataset_version': DATASET_VERSION,
                'dataset_path': str(COOKED / 'step_03_features' / 'features_engineered.pkl'),
                'script': 'pipeline/00_scripts/phaseC_lp_test.py',
            },
            stability=stability,
            meta={'baseline_unweighted_stability': stability_baseline, 'pretrigger': PRETRIGGER},
        )

if __name__ == '__main__':
    main()
