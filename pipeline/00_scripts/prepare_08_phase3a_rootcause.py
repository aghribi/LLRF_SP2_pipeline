#!/usr/bin/env python3
"""
Step 08: Phase 3A - Root Cause Identification

Description: Identify PRIMARY fault among multiple triggers using temporal analysis.

Input:  cooked_data/step_03_features/features_engineered.pkl
Output: cooked_data/step_08_phase3a/root_cause.pkl
"""

import sys
import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utilities.reporting.manifest import save_manifest

sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import leakage_safe_split, repeated_leakage_safe_eval
from split_diagnostics import log_split_composition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler('step_08_phase3a.log'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


class Step08Processor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_inputs(self):
        with open(self.input_dir / 'features_engineered.pkl', 'rb') as f:
            self.data = pickle.load(f)
        logger.info(f"Loaded: {self.data['features_all'].shape}, "
                    f"{len(self.data['feature_cols'])} filtered feature columns")

    def process(self):
        logger.info("="*70)
        logger.info("PHASE 3A: ROOT CAUSE IDENTIFICATION")
        logger.info("="*70)

        y_multilabel = self.data['y_multilabel']

        if y_multilabel is None:
            raise ValueError("No multi-label data")

        # For root cause, identify PRIMARY fault (first active bit in ALM)
        # Simplified: take argmax of multi-hot encoding
        fault_indices = np.argmax(y_multilabel, axis=1)
        has_fault = y_multilabel.sum(axis=1) > 0

        n_faults = int(np.sum(has_fault))
        if n_faults == 0:
            logger.warning("No fault events found")
            return {}

        logger.info(f"Root cause analysis on {n_faults} fault events")

        # Subset to fault events FIRST, then split -- scaler/PCA are fit on
        # the train fold of that subset only (leakage-safe).
        fault_data = {
            'features_all': self.data['features_all'].loc[has_fault].reset_index(drop=True),
            'feature_cols': self.data['feature_cols'],
        }
        y_root = fault_indices[has_fault]
        split = leakage_safe_split(fault_data, y_root, test_size=0.3, random_state=42,
                                    stratify=False)
        X_train, X_test = split['X_train'], split['X_test']
        y_train, y_test = split['y_train'], split['y_test']
        log_split_composition(y_train, y_test, self.data['fault_column_names'], logger,
                               title='Step 08 (root cause) split composition')

        # Random Forest for root cause classification
        # Phase 4 (2026-09-06): class_weight='balanced' for the multiclass case
        # -- sklearn computes balanced weights from y_train's own class counts
        # at fit time.
        logger.info("\nTraining Root Cause Classifier (class_weight='balanced')...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        logger.info("\nRoot Cause Classification:")
        # Get unique labels present in test set
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        target_names = [self.data['fault_column_names'][i] for i in unique_labels]
        logger.info(f"{classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names)}")

        headline_accuracy = accuracy_score(y_test, y_pred)
        headline_macro_f1 = f1_score(y_test, y_pred, labels=unique_labels, average='macro', zero_division=0)
        headline_weighted_f1 = f1_score(y_test, y_pred, labels=unique_labels, average='weighted', zero_division=0)
        logger.info(f"Headline (single split, seed=42): accuracy={headline_accuracy:.4f} "
                    f"macro_f1={headline_macro_f1:.4f} weighted_f1={headline_weighted_f1:.4f}")

        # Repeated-split stability check: this script's own "ground truth" is
        # np.argmax(y_multilabel) -- a trivial deterministic rule (see plan
        # Phase 3) -- so at minimum its reported accuracy should be checked
        # against split noise before citing it as an ML result.
        # Phase 4 (2026-09-06): runs the pre-weighting baseline alongside
        # class_weight='balanced' so the effect is measured, not assumed.
        logger.info("\nRepeated-split stability check (10 splits, different random_state each): "
                    "baseline (unweighted) vs. class_weight='balanced'...")
        all_labels = list(range(len(self.data['fault_column_names'])))
        label_names = self.data['fault_column_names']

        def model_fn_baseline(X_train, y_train):
            m = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            m.fit(X_train, y_train)
            return m

        def model_fn_balanced(X_train, y_train):
            m = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
            m.fit(X_train, y_train)
            return m

        # f1_score/accuracy_score already imported at module level (line 18);
        # a redundant local import here previously shadowed them for this
        # whole function, causing headline_accuracy = accuracy_score(...)
        # above to raise UnboundLocalError.
        metric_fns = {
            'accuracy': lambda yt, yp: accuracy_score(yt, yp),
            'macro_f1': lambda yt, yp: f1_score(yt, yp, labels=all_labels, average='macro', zero_division=0),
            'per_label_f1': lambda yt, yp: f1_score(yt, yp, labels=all_labels, average=None, zero_division=0),
        }
        stability_baseline = repeated_leakage_safe_eval(
            fault_data, y_root, model_fn_baseline, metric_fns, n_repeats=10,
            test_size=0.3, stratify=False, base_random_state=0, label_names=label_names,
        )
        stability = repeated_leakage_safe_eval(
            fault_data, y_root, model_fn_balanced, metric_fns, n_repeats=10,
            test_size=0.3, stratify=False, base_random_state=0, label_names=label_names,
        )
        logger.info(f"  accuracy: baseline mean={stability_baseline['accuracy']['mean']:.4f} -> "
                    f"balanced mean={stability['accuracy']['mean']:.4f}")
        logger.info(f"  macro_f1: baseline mean={stability_baseline['macro_f1']['mean']:.4f} -> "
                    f"balanced mean={stability['macro_f1']['mean']:.4f}")
        logger.info("  Per-label F1, baseline -> balanced:")
        for lname in label_names:
            b = stability_baseline['per_label_f1']['per_label'][lname]
            a = stability['per_label_f1']['per_label'][lname]
            logger.info(f"    [{lname}]: {b['mean']:.3f}+/-{b['std']:.3f} -> {a['mean']:.3f}+/-{a['std']:.3f}")

        save_manifest(
            phase='08_phase3a_rootcause_stability',
            metrics={
                'single_split_n_test': {'value': int(len(y_test)), 'fmt': None, 'label': 'Single-split (seed=42) test set size'},
            },
            pipeline_run={
                'dataset_version': 'V6',
                'dataset_path': str(self.input_dir / 'features_engineered.pkl'),
                'script': 'pipeline/00_scripts/prepare_08_phase3a_rootcause.py',
            },
            stability=stability,
            meta={
                'baseline_unweighted_stability': stability_baseline,
                'ground_truth_caveat': (
                    "y_root is np.argmax(y_multilabel, axis=1) -- the first flagged "
                    "category in bit order, itself a deterministic rule this RF is "
                    "trained to approximate. See plan Phase 3 (root cause task design)."
                ),
            },
        )

        # Canonical headline manifest -- 2026-09-06. The single-split numbers
        # here are what a report would naturally want to cite; the repeated-
        # split mean/std above (08_phase3a_rootcause_stability.yaml) is the
        # more honest number and should be preferred/cited alongside this one,
        # not instead of it -- see meta.stability_cross_reference below.
        save_manifest(
            phase='08_phase3a_rootcause',
            metrics={
                'headline_accuracy': {'value': float(headline_accuracy), 'fmt': '.1%',
                                       'label': 'Single-split (seed=42) accuracy'},
                'headline_macro_f1': {'value': float(headline_macro_f1), 'fmt': '.3f',
                                       'label': 'Single-split (seed=42) macro-F1'},
                'headline_weighted_f1': {'value': float(headline_weighted_f1), 'fmt': '.3f',
                                          'label': 'Single-split (seed=42) weighted-F1'},
                'n_test': {'value': int(len(y_test)), 'fmt': None, 'label': 'Test-set events'},
                'n_fault_events': {'value': int(n_faults), 'fmt': None,
                                    'label': 'Total fault events used (root-cause task, fault-only subset)'},
            },
            pipeline_run={
                'dataset_version': 'V6',
                'dataset_path': str(self.input_dir / 'features_engineered.pkl'),
                'script': 'pipeline/00_scripts/prepare_08_phase3a_rootcause.py',
            },
            meta={
                'stability_cross_reference': (
                    "See 08_phase3a_rootcause_stability.yaml for the repeated-split (10x) "
                    f"mean/std: class_weight='balanced' accuracy={stability['accuracy']['mean']:.4f}, "
                    f"macro_f1={stability['macro_f1']['mean']:.4f} -- prefer citing that mean/std "
                    "over this single-split number where both fit."
                ),
                'ground_truth_caveat': (
                    "y_root is np.argmax(y_multilabel, axis=1) -- the first flagged category in bit "
                    "order, itself a deterministic rule this RF is trained to approximate. Real ML "
                    "value here would require independent root-cause labels for cases where 'first "
                    "bit set' is not the true physical cause; none currently exist. See "
                    "PHASE3_WEAKNESS_DIAGNOSIS_AND_OPERATIONAL_USABILITY.md section 2.2."
                ),
            },
        )

        return {
            'model': clf,
            'predictions': y_pred,
            'data_splits': {
                'X_test': X_test, 'y_test': y_test,
                'idx_train': split['idx_train'], 'idx_test': split['idx_test'],
                'scaler': split['scaler'], 'pca': split['pca'],
                'feature_cols': split['feature_cols'],
            },
            'fault_names': self.data['fault_column_names']
        }

    def save_outputs(self, results):
        output_file = self.output_dir / 'root_cause.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f, protocol=4)
        logger.info(f"\n✓ Saved: {output_file}")

    def run(self):
        logger.info("STEP 08: PHASE 3A - ROOT CAUSE IDENTIFICATION")
        self.load_inputs()
        results = self.process()
        self.save_outputs(results)
        logger.info("STEP 08 COMPLETE!")
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    Step08Processor(args.input, args.output).run()


if __name__ == '__main__':
    main()
