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
from pathlib import Path
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, classification_report, f1_score
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(features_file, max_samples=None):
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
    X = data.get('X_scaled')
    y = data.get('y_multilabel')

    if X is None or y is None:
        raise ValueError('features_engineered.pkl missing X_scaled or y_multilabel')

    if max_samples is not None and max_samples < X.shape[0]:
        idx = np.random.RandomState(42).choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    return X, y, data


def train_br_rf(X_train, y_train, X_test):
    clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred


def train_cc_xgb(X_train, y_train, X_test):
    if not XGB_AVAILABLE:
        logger.warning('XGBoost not available — skipping classifier chain')
        return None, None
    from sklearn.multioutput import ClassifierChain
    base = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', n_jobs=4)
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
    parser.add_argument('--max-samples', type=int, default=1000)
    args = parser.parse_args()

    features_file = Path(args.features)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, data = load_data(features_file, max_samples=args.max_samples)
    label_names = data.get('fault_column_names')

    logger.info(f'Loaded data: X={X.shape}, y={y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 1) Binary Relevance (MultiOutput RF)
    logger.info('Training BR MultiOutput RandomForest...')
    br_model, br_pred = train_br_rf(X_train, y_train, X_test)
    br_results = evaluate(y_test, br_pred, label_names)
    with open(out_dir / 'br_results.pkl', 'wb') as f:
        pickle.dump({'model': br_model, 'results': br_results}, f)
    logger.info(f'BR results: hamming={br_results["hamming_loss"]:.4f} micro_f1={br_results["micro_f1"]:.4f} macro_f1={br_results["macro_f1"]:.4f}')

    # 2) Optional: Classifier Chain with XGBoost
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
        'cc': cc_results if XGB_AVAILABLE else None,
        'label_names': label_names
    }
    with open(out_dir / 'phaseA_summary.pkl', 'wb') as f:
        pickle.dump(summary, f)

    logger.info('Phase A test complete.')


if __name__ == '__main__':
    main()
