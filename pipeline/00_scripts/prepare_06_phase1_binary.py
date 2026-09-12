#!/usr/bin/env python3
"""
Step 06: Phase 1 - Binary Classification

Description: Fault vs No-Fault classification using multiple algorithms.
             Optimized for production deployment and real-time fault detection.

Input:  cooked_data/step_03_features/features_engineered.pkl
Output: cooked_data/step_06_phase1/binary_classification.pkl

Models:
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- SVM (RBF kernel)

Usage:
    python prepare_06_phase1_binary.py \
        --input /path/to/step_03_features \
        --output /path/to/step_06_phase1
"""

import sys
import logging
from pathlib import Path
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix,
                              accuracy_score, precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utilities.reporting.manifest import save_manifest

sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import leakage_safe_split, repeated_leakage_safe_eval
from split_diagnostics import log_split_composition

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step_06_phase1.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Step06Processor:
    """Processor for Step 06: Binary Classification"""

    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_data = None

    def load_inputs(self):
        """Load feature data"""
        logger.info("Loading features from Step 03...")

        features_file = self.input_dir / 'features_engineered.pkl'
        with open(features_file, 'rb') as f:
            self.feature_data = pickle.load(f)

        logger.info(f"Loaded: {self.feature_data['features_all'].shape}, "
                    f"{len(self.feature_data['feature_cols'])} filtered feature columns")

    def process(self):
        """Train binary classification models"""
        logger.info("="*70)
        logger.info("PHASE 1: BINARY CLASSIFICATION")
        logger.info("="*70)

        y = self.feature_data['y_binary']

        if y is None:
            raise ValueError("No binary labels available")

        n_events = len(y)
        logger.info(f"Dataset: {n_events} events")
        logger.info(f"  Normal: {np.sum(y == 0)}, Fault: {np.sum(y == 1)}")

        # Leakage-safe split: scaler/PCA are fit on the TRAIN fold only, not on
        # the full dataset (see leakage_safe_features.py for why this matters).
        split = leakage_safe_split(self.feature_data, y, test_size=0.3, random_state=42)
        X_train, X_test = split['X_train'], split['X_test']
        y_train, y_test = split['y_train'], split['y_test']
        logger.info(f"Features after train-fold-fit scaling+PCA: {X_train.shape[1]}")
        log_split_composition(y_train, y_test, ['Normal', 'Fault'], logger,
                               title='Step 06 (binary classification) split composition')

        results = {}

        # ========================================
        # 1. Logistic Regression
        # ========================================
        logger.info("\nTraining Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        lr.fit(X_train, y_train)

        y_pred_lr = lr.predict(X_test)
        y_proba_lr = lr.predict_proba(X_test)[:, 1]

        logger.info("Logistic Regression:")
        logger.info(f"\n{classification_report(y_test, y_pred_lr)}")
        logger.info(f"ROC AUC: {roc_auc_score(y_test, y_proba_lr):.3f}")

        results['logistic_regression'] = {
            'model': lr,
            'predictions': y_pred_lr,
            'probabilities': y_proba_lr,
            'roc_auc': roc_auc_score(y_test, y_proba_lr)
        }

        # ========================================
        # 2. Random Forest
        # ========================================
        logger.info("\nTraining Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        y_pred_rf = rf.predict(X_test)
        y_proba_rf = rf.predict_proba(X_test)[:, 1]

        logger.info("Random Forest:")
        logger.info(f"\n{classification_report(y_test, y_pred_rf)}")
        logger.info(f"ROC AUC: {roc_auc_score(y_test, y_proba_rf):.3f}")

        results['random_forest'] = {
            'model': rf,
            'predictions': y_pred_rf,
            'probabilities': y_proba_rf,
            'roc_auc': roc_auc_score(y_test, y_proba_rf),
            'feature_importances': rf.feature_importances_
        }

        # ========================================
        # 3. XGBoost
        # ========================================
        if XGBOOST_AVAILABLE:
            logger.info("\nTraining XGBoost...")
            xgb = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )
            xgb.fit(X_train, y_train)

            y_pred_xgb = xgb.predict(X_test)
            y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

            logger.info("XGBoost:")
            logger.info(f"\n{classification_report(y_test, y_pred_xgb)}")
            logger.info(f"ROC AUC: {roc_auc_score(y_test, y_proba_xgb):.3f}")

            results['xgboost'] = {
                'model': xgb,
                'predictions': y_pred_xgb,
                'probabilities': y_proba_xgb,
                'roc_auc': roc_auc_score(y_test, y_proba_xgb),
                'feature_importances': xgb.feature_importances_
            }
        else:
            logger.warning("XGBoost not available - skipping")

        # ========================================
        # 4. SVM
        # ========================================
        logger.info("\nTraining SVM (may take longer)...")
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        svm.fit(X_train, y_train)

        y_pred_svm = svm.predict(X_test)
        y_proba_svm = svm.predict_proba(X_test)[:, 1]

        logger.info("SVM:")
        logger.info(f"\n{classification_report(y_test, y_pred_svm)}")
        logger.info(f"ROC AUC: {roc_auc_score(y_test, y_proba_svm):.3f}")

        results['svm'] = {
            'model': svm,
            'predictions': y_pred_svm,
            'probabilities': y_proba_svm,
            'roc_auc': roc_auc_score(y_test, y_proba_svm)
        }

        # Single-split per-model comparison manifest (V6 fix, 2026-09-07):
        # this used to come from a notebook snapshot (analysis/notebooks/
        # 06_binary_classification.ipynb) that predates the whole-signal
        # mean/std/rms/energy normalization-order fix in
        # prepare_data_cluster_v6.py (see that script's docstring) and was
        # therefore silently citing a stale n_features=770 (now 777) plus
        # metrics computed against the pre-fix feature set. Saved directly
        # from this script's own live run instead, so it can never drift
        # out of sync with the actual pipeline again.
        model_key_map = {
            'logistic_regression': 'logistic_regression',
            'random_forest': 'random_forest',
            'xgboost': 'xgboost',
            'svm': 'svm_rbf',
        }
        comparison_metrics = {}
        for internal_name, macro_slug in model_key_map.items():
            if internal_name not in results:
                continue
            y_pred = results[internal_name]['predictions']
            y_proba = results[internal_name]['probabilities']
            label = {'logistic_regression': 'Logistic Regression', 'random_forest': 'Random Forest',
                      'xgboost': 'XGBoost', 'svm': 'SVM (RBF)'}[internal_name]
            comparison_metrics[f'{macro_slug}_accuracy'] = {
                'value': float(accuracy_score(y_test, y_pred)), 'fmt': '.1%', 'label': f'{label} accuracy'}
            comparison_metrics[f'{macro_slug}_roc_auc'] = {
                'value': float(roc_auc_score(y_test, y_proba)), 'fmt': '.4f', 'label': f'{label} ROC AUC'}
            comparison_metrics[f'{macro_slug}_precision'] = {
                'value': float(precision_score(y_test, y_pred, zero_division=0)), 'fmt': '.1%', 'label': f'{label} precision'}
            comparison_metrics[f'{macro_slug}_recall'] = {
                'value': float(recall_score(y_test, y_pred, zero_division=0)), 'fmt': '.1%', 'label': f'{label} recall'}
            comparison_metrics[f'{macro_slug}_f1'] = {
                'value': float(f1_score(y_test, y_pred, zero_division=0)), 'fmt': '.3f', 'label': f'{label} F1'}
        comparison_metrics['n_events_total'] = {'value': int(n_events), 'fmt': None, 'label': 'Total events'}
        comparison_metrics['n_events_test'] = {'value': int(len(y_test)), 'fmt': None, 'label': 'Test-set events'}
        comparison_metrics['n_features'] = {'value': int(X_train.shape[1]), 'fmt': None, 'label': 'Features used'}

        save_manifest(
            phase='06_binary_classification',
            metrics=comparison_metrics,
            pipeline_run={
                'dataset_version': (lambda _n: _n.upper() if _n else 'V2')(self.input_dir.name.replace('cooked_data', '').lstrip('_')),
                'dataset_path': str(self.input_dir / 'features_engineered.pkl'),
                'script': 'pipeline/00_scripts/prepare_06_phase1_binary.py',
            },
            meta={
                'leakage_investigation': (
                    'See spiral2_open_methodology_questions memory item 5 and '
                    'NOTEBOOK_AUDIT_2026-09-03.md for the full root-cause chain '
                    'and the raw-signal verification that this result is genuine.'
                ),
            },
        )

        # Repeated-split stability check for the two models the report cites
        # as headline results (RF, XGBoost) -- is the near-100% separability
        # a single-split artifact or does it hold across splits? (2026-09-05
        # audit: no script anywhere previously re-ran its split more than once.)
        logger.info("\nRepeated-split stability check (10 splits, different random_state each)...")

        metric_fns = {
            'accuracy': lambda yt, yp: accuracy_score(yt, yp),
            'f1': lambda yt, yp: f1_score(yt, yp, zero_division=0),
        }
        stability_all = {}

        def rf_model_fn(X_train, y_train):
            m = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            m.fit(X_train, y_train)
            return m

        stability_all['random_forest'] = repeated_leakage_safe_eval(
            self.feature_data, y, rf_model_fn, metric_fns, n_repeats=10,
            test_size=0.3, stratify=True, base_random_state=0,
        )
        logger.info(f"  RF accuracy: mean={stability_all['random_forest']['accuracy']['mean']:.4f} "
                    f"std={stability_all['random_forest']['accuracy']['std']:.4f}")

        if XGBOOST_AVAILABLE:
            def xgb_model_fn(X_train, y_train):
                m = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
                m.fit(X_train, y_train)
                return m

            stability_all['xgboost'] = repeated_leakage_safe_eval(
                self.feature_data, y, xgb_model_fn, metric_fns, n_repeats=10,
                test_size=0.3, stratify=True, base_random_state=0,
            )
            logger.info(f"  XGBoost accuracy: mean={stability_all['xgboost']['accuracy']['mean']:.4f} "
                        f"std={stability_all['xgboost']['accuracy']['std']:.4f}")

        save_manifest(
            phase='06_phase1_binary_stability',
            metrics={
                'single_split_rf_roc_auc': {'value': float(results['random_forest']['roc_auc']), 'fmt': '.4f',
                                             'label': 'Single-split (seed=42) RF ROC AUC'},
            },
            pipeline_run={
                'dataset_version': (lambda _n: _n.upper() if _n else 'V2')(self.input_dir.name.replace('cooked_data', '').lstrip('_')),
                'dataset_path': str(self.input_dir / 'features_engineered.pkl'),
                'script': 'pipeline/00_scripts/prepare_06_phase1_binary.py',
            },
            stability=stability_all['random_forest'],
            meta={'xgboost_stability': stability_all.get('xgboost')},
        )

        # Save splits, plus the train-fold-fit scaler/PCA (needed by any
        # downstream script, e.g. SHAP analysis, that must transform new rows
        # the same way these models were trained -- never refit on test data).
        results['data_splits'] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'idx_train': split['idx_train'],
            'idx_test': split['idx_test'],
            'scaler': split['scaler'],
            'pca': split['pca'],
            'feature_cols': split['feature_cols'],
        }

        return results

    def save_outputs(self, results):
        """Save results"""
        output_file = self.output_dir / 'binary_classification.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f, protocol=4)

        logger.info(f"\n✓ Saved: {output_file} ({output_file.stat().st_size/(1024**2):.2f} MB)")

    def run(self):
        """Execute pipeline"""
        logger.info("="*70)
        logger.info("STEP 06: PHASE 1 - BINARY CLASSIFICATION")
        logger.info("="*70)

        self.load_inputs()
        results = self.process()
        self.save_outputs(results)

        logger.info("\n" + "="*70)
        logger.info("STEP 06 COMPLETE!")
        logger.info("="*70)

        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Step 06: Binary Classification')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    processor = Step06Processor(args.input, args.output)
    processor.run()


if __name__ == '__main__':
    main()
