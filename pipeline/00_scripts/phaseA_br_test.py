#!/usr/bin/env python3
"""
Phase A — Binary Relevance baseline test script
- Loads `features_engineered.pkl`
- Subsamples a small number of events (for quick iteration)
- Trains MultiOutputClassifier(RandomForest) and a ClassifierChain(XGBoost if available)
- Prints evaluation metrics and saves results to output dir
"""

import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, classification_report, f1_score
import logging
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utilities.reporting.manifest import save_manifest

sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import leakage_safe_split, repeated_leakage_safe_eval, compute_scale_pos_weight, true_precursor_columns
from split_diagnostics import log_split_composition

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(features_file, max_samples=None, pretrigger=False):
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
    y = data.get('y_multilabel')

    if data.get('features_all') is None or y is None:
        raise ValueError('features_engineered.pkl missing features_all or y_multilabel')

    n_total = len(y)
    sample_idx = np.arange(n_total)
    if max_samples is not None and max_samples < n_total:
        sample_idx = np.random.RandomState(42).choice(n_total, size=max_samples, replace=False)
        y = y[sample_idx]

    feature_cols = data['feature_cols']
    if pretrigger:
        # Same restriction as prepare_05_phase0_precursor.py / step 06b -- see
        # leakage_safe_features.true_precursor_columns() for why the full
        # feature set isn't appropriate for a pre-trigger-only task.
        feature_cols = true_precursor_columns(feature_cols)

    # Leakage-safe: subsample features_all rows to match y, keep as pkl-shaped
    # dict so leakage_safe_split can fit scaler/PCA on the train fold only.
    sampled_data = {
        'features_all': data['features_all'].iloc[sample_idx].reset_index(drop=True),
        'feature_cols': feature_cols,
    }

    return sampled_data, y, data


def train_br_rf(X_train, y_train, X_test):
    # Phase 4 (2026-09-06): class_weight='balanced' -- MultiOutputClassifier
    # clones+refits per label column, so this is correctly per-label balanced.
    clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=42,
                                                        class_weight='balanced', n_jobs=-1))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred


def train_cc_xgb(X_train, y_train, X_test):
    if not XGB_AVAILABLE:
        logger.warning('XGBoost not available — skipping classifier chain')
        return None, None
    from sklearn.multioutput import ClassifierChain
    # Phase 4 (2026-09-06): scale_pos_weight, data-driven from y_train's
    # aggregate imbalance. See compute_scale_pos_weight()'s docstring for why
    # this is a single value applied across the whole chain, not truly
    # per-label like sklearn's class_weight='balanced' above.
    spw = compute_scale_pos_weight(y_train)
    base = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss',
                          scale_pos_weight=spw, n_jobs=4)
    chain = ClassifierChain(base, order='random', random_state=42)
    chain.fit(X_train, y_train)
    y_pred = chain.predict(X_test)
    return chain, y_pred


def evaluate(y_test, y_pred, label_names=None):
    if y_pred is None:
        return {}
    results = {}
    results['hamming_loss'] = hamming_loss(y_test, y_pred)
    results['micro_f1'] = f1_score(y_test, y_pred, average='micro', zero_division=0)
    results['macro_f1'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
    # Per-label reports
    if label_names is None:
        label_names = [f'Fault_{i}' for i in range(y_test.shape[1])]

    per_label = {}
    for i, name in enumerate(label_names):
        per_label[name] = classification_report(y_test[:, i], y_pred[:, i], output_dict=True, zero_division=0)
    results['per_label'] = per_label
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--pretrigger', action='store_true',
                         help='Restrict to the pre-trigger-only true-precursor feature subset')
    args = parser.parse_args()

    features_file = Path(args.features)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_suffix = '_pretrigger' if args.pretrigger else ''
    dataset_version = (lambda _n: _n.upper() if _n else 'V2')(
        features_file.parent.name.replace('cooked_data', '').lstrip('_'))

    sampled_data, y, data = load_data(features_file, max_samples=args.max_samples, pretrigger=args.pretrigger)
    label_names = data.get('fault_column_names')

    logger.info(f'Loaded data: y={y.shape}')

    # Leakage-safe split: scaler/PCA fit on the TRAIN fold only. y is
    # multi-label (2D) so stratify is off, matching the original code.
    split = leakage_safe_split(sampled_data, y, test_size=0.3, random_state=42,
                                stratify=False)
    X_train, X_test = split['X_train'], split['X_test']
    y_train, y_test = split['y_train'], split['y_test']
    log_split_composition(y_train, y_test, label_names, logger, title='Phase A (BR/CC) split composition')

    # 1) Binary Relevance (MultiOutput RF)
    logger.info('Training BR MultiOutput RandomForest...')
    br_model, br_pred = train_br_rf(X_train, y_train, X_test)
    br_results = evaluate(y_test, br_pred, label_names)
    with open(out_dir / 'br_results.pkl', 'wb') as f:
        pickle.dump({
            'model': br_model, 'results': br_results,
            'scaler': split['scaler'], 'pca': split['pca'],
            'feature_cols': split['feature_cols'],
            'idx_train': split['idx_train'], 'idx_test': split['idx_test'],
        }, f)
    logger.info(f'BR results: hamming={br_results["hamming_loss"]:.4f} micro_f1={br_results["micro_f1"]:.4f} macro_f1={br_results["macro_f1"]:.4f}')

    # 2) Optional: Classifier Chain with XGBoost
    cc_results = None
    if XGB_AVAILABLE:
        logger.info('Training ClassifierChain XGBoost...')
        cc_model, cc_pred = train_cc_xgb(X_train, y_train, X_test)
        cc_results = evaluate(y_test, cc_pred, label_names)
        with open(out_dir / 'cc_results.pkl', 'wb') as f:
            pickle.dump({'model': cc_model, 'results': cc_results}, f)
        logger.info(f'CC results: hamming={cc_results["hamming_loss"]:.4f} micro_f1={cc_results["micro_f1"]:.4f} macro_f1={cc_results["macro_f1"]:.4f}')
    else:
        logger.info('XGBoost not available — skipped CC')

    # Save a small summary
    summary = {
        'br': br_results,
        'cc': cc_results,
        'label_names': label_names
    }
    with open(out_dir / 'phaseA_summary.pkl', 'wb') as f:
        pickle.dump(summary, f)

    # Repeated-split stability check: is this single 70/30 split's macro-F1
    # a stable estimate, or noise? (2026-09-05 audit finding.)
    # Phase 4 (2026-09-06): also runs the pre-weighting baseline alongside the
    # class_weight='balanced' (BR) / scale_pos_weight (CC) variants, so the
    # effect is measured, not assumed.
    logger.info("Repeated-split stability check (10 splits, different random_state each): "
                "baseline (unweighted) vs. class-weighted...")

    def br_model_fn_baseline(X_train, y_train):
        m = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
        m.fit(X_train, y_train)
        return m

    def br_model_fn_balanced(X_train, y_train):
        m = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=42,
                                                          class_weight='balanced', n_jobs=-1))
        m.fit(X_train, y_train)
        return m

    metric_fns = {
        'macro_f1': lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
        'per_label_f1': lambda yt, yp: f1_score(yt, yp, average=None, zero_division=0),
        'hamming_loss': lambda yt, yp: hamming_loss(yt, yp),
    }
    br_stability_baseline = repeated_leakage_safe_eval(
        sampled_data, y, br_model_fn_baseline, metric_fns, n_repeats=10,
        test_size=0.3, stratify=False, base_random_state=0, label_names=label_names,
    )
    br_stability = repeated_leakage_safe_eval(
        sampled_data, y, br_model_fn_balanced, metric_fns, n_repeats=10,
        test_size=0.3, stratify=False, base_random_state=0, label_names=label_names,
    )
    logger.info(f"  BR macro_f1: baseline mean={br_stability_baseline['macro_f1']['mean']:.4f} -> "
                f"balanced mean={br_stability['macro_f1']['mean']:.4f}")
    logger.info("  BR per-label F1, baseline -> balanced:")
    for lname in label_names:
        b = br_stability_baseline['per_label_f1']['per_label'][lname]
        a = br_stability['per_label_f1']['per_label'][lname]
        logger.info(f"    [{lname}]: {b['mean']:.3f}+/-{b['std']:.3f} -> {a['mean']:.3f}+/-{a['std']:.3f}")

    stability_metrics = {'br': br_stability}
    stability_baselines = {'br': br_stability_baseline}
    if XGB_AVAILABLE:
        def cc_model_fn_baseline(X_train, y_train):
            from sklearn.multioutput import ClassifierChain
            base = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', n_jobs=4)
            m = ClassifierChain(base, order='random', random_state=42)
            m.fit(X_train, y_train)
            return m

        def cc_model_fn_balanced(X_train, y_train):
            from sklearn.multioutput import ClassifierChain
            spw = compute_scale_pos_weight(y_train)
            base = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss',
                                  scale_pos_weight=spw, n_jobs=4)
            m = ClassifierChain(base, order='random', random_state=42)
            m.fit(X_train, y_train)
            return m

        cc_stability_baseline = repeated_leakage_safe_eval(
            sampled_data, y, cc_model_fn_baseline, metric_fns, n_repeats=10,
            test_size=0.3, stratify=False, base_random_state=0, label_names=label_names,
        )
        cc_stability = repeated_leakage_safe_eval(
            sampled_data, y, cc_model_fn_balanced, metric_fns, n_repeats=10,
            test_size=0.3, stratify=False, base_random_state=0, label_names=label_names,
        )
        logger.info(f"  CC macro_f1: baseline mean={cc_stability_baseline['macro_f1']['mean']:.4f} -> "
                    f"weighted mean={cc_stability['macro_f1']['mean']:.4f}")
        logger.info("  CC per-label F1, baseline -> weighted:")
        for lname in label_names:
            b = cc_stability_baseline['per_label_f1']['per_label'][lname]
            a = cc_stability['per_label_f1']['per_label'][lname]
            logger.info(f"    [{lname}]: {b['mean']:.3f}+/-{b['std']:.3f} -> {a['mean']:.3f}+/-{a['std']:.3f}")
        stability_metrics['cc'] = cc_stability
        stability_baselines['cc'] = cc_stability_baseline

    save_manifest(
        phase=f'07A_phaseA_br_cc_stability{phase_suffix}',
        metrics={
            'single_split_br_macro_f1': {'value': float(br_results['macro_f1']), 'fmt': '.4f', 'label': 'Single-split (seed=42) BR macro-F1'},
        },
        pipeline_run={
            'dataset_version': dataset_version,
            'dataset_path': str(features_file),
            'script': 'pipeline/00_scripts/phaseA_br_test.py',
        },
        stability=stability_metrics['br'],
        meta={
            'baseline_unweighted_stability': stability_baselines['br'],
            'cc_stability_note': 'ClassifierChain (cc) stability stored separately -- see phase 07A_phaseA_cc_stability.' if 'cc' in stability_metrics else None,
            'pretrigger': args.pretrigger,
        },
    )
    if 'cc' in stability_metrics:
        save_manifest(
            phase=f'07A_phaseA_cc_stability{phase_suffix}',
            metrics={
                'single_split_cc_macro_f1': {'value': float(cc_results['macro_f1']), 'fmt': '.4f', 'label': 'Single-split (seed=42) CC macro-F1'},
            },
            pipeline_run={
                'dataset_version': dataset_version,
                'dataset_path': str(features_file),
                'script': 'pipeline/00_scripts/phaseA_br_test.py',
            },
            stability=stability_metrics['cc'],
            meta={'baseline_unweighted_stability': stability_baselines['cc'], 'pretrigger': args.pretrigger},
        )

    logger.info('Phase A test complete.')


if __name__ == '__main__':
    main()
