#!/usr/bin/env python3
"""
Step 07: Phase 2 - Multi-Label Classification

Description: Classify 7 ALM fault types (can have multiple simultaneous faults).

Input:  cooked_data/step_03_features/features_engineered.pkl
Output: cooked_data/step_07_phase2/multilabel_triggers.pkl
"""

import sys
import logging
from pathlib import Path
import pickle
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, hamming_loss
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utilities.reporting.manifest import save_manifest

sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import leakage_safe_split, repeated_leakage_safe_eval
from split_diagnostics import log_split_composition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler('step_07_phase2.log'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


class Step07Processor:
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
        logger.info("PHASE 2: MULTI-LABEL CLASSIFICATION")
        logger.info("="*70)

        y = self.data['y_multilabel']

        if y is None:
            raise ValueError("No multi-label data available")

        logger.info(f"Training on {y.shape[0]} events, {y.shape[1]} fault types")

        # Leakage-safe split: scaler/PCA fit on the TRAIN fold only. y is
        # multi-label (2D) so stratify is off, matching the original code's
        # unstratified split.
        split = leakage_safe_split(self.data, y, test_size=0.3, random_state=42,
                                    stratify=False)
        X_train, X_test = split['X_train'], split['X_test']
        y_train, y_test = split['y_train'], split['y_test']
        log_split_composition(y_train, y_test, self.data['fault_column_names'], logger,
                               title='Step 07 (MultiOutput RF baseline) split composition')

        # Multi-output Random Forest
        # Phase 4 (2026-09-06): class_weight='balanced' -- MultiOutputClassifier
        # clones the base estimator and re-fits independently per label column,
        # so sklearn recomputes 'balanced' weights from each column's own
        # class counts at fit time -- this is correctly PER-LABEL balancing,
        # not a single global value (unlike XGBoost's scale_pos_weight, which
        # has no such auto-recompute and must be set explicitly per model).
        logger.info("\nTraining Multi-Output Random Forest (class_weight='balanced')...")
        clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42,
                                                            class_weight='balanced', n_jobs=-1))
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        logger.info("\nPer-Label Performance:")
        for i, fault_name in enumerate(self.data['fault_column_names']):
            logger.info(f"\n{fault_name}:")
            logger.info(f"{classification_report(y_test[:, i], y_pred[:, i])}")

        hamming = hamming_loss(y_test, y_pred)
        logger.info(f"\nHamming Loss: {hamming:.4f}")

        # Repeated-split stability check: is this single 70/30 split's macro-F1
        # a stable estimate, or noise? (2026-09-05 audit: no script anywhere
        # previously re-ran its split more than once.)
        # Phase 4 (2026-09-06): also runs the PRE-class-weighting baseline
        # alongside the class_weight='balanced' variant, so the effect of
        # weighting is measured, not assumed -- per the plan's own
        # verification criterion.
        logger.info("\nRepeated-split stability check (10 splits, different random_state each): "
                    "baseline (unweighted) vs. class_weight='balanced'...")
        label_names = self.data['fault_column_names']

        def model_fn_baseline(X_train, y_train):
            m = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
            m.fit(X_train, y_train)
            return m

        def model_fn_balanced(X_train, y_train):
            m = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42,
                                                              class_weight='balanced', n_jobs=-1))
            m.fit(X_train, y_train)
            return m

        from sklearn.metrics import f1_score
        metric_fns = {
            'macro_f1': lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
            'per_label_f1': lambda yt, yp: f1_score(yt, yp, average=None, zero_division=0),
            'hamming_loss': lambda yt, yp: hamming_loss(yt, yp),
        }
        stability_baseline = repeated_leakage_safe_eval(
            self.data, y, model_fn_baseline, metric_fns, n_repeats=10,
            test_size=0.3, stratify=False, base_random_state=0, label_names=label_names,
        )
        stability = repeated_leakage_safe_eval(
            self.data, y, model_fn_balanced, metric_fns, n_repeats=10,
            test_size=0.3, stratify=False, base_random_state=0, label_names=label_names,
        )
        logger.info(f"  macro_f1: baseline mean={stability_baseline['macro_f1']['mean']:.4f} -> "
                    f"balanced mean={stability['macro_f1']['mean']:.4f} "
                    f"(std {stability_baseline['macro_f1']['std']:.4f} -> {stability['macro_f1']['std']:.4f})")
        logger.info(f"  hamming_loss: baseline mean={stability_baseline['hamming_loss']['mean']:.4f} -> "
                    f"balanced mean={stability['hamming_loss']['mean']:.4f}")
        logger.info("  Per-label F1, baseline -> balanced:")
        for lname in label_names:
            b = stability_baseline['per_label_f1']['per_label'][lname]
            a = stability['per_label_f1']['per_label'][lname]
            logger.info(f"    [{lname}]: {b['mean']:.3f}+/-{b['std']:.3f} -> {a['mean']:.3f}+/-{a['std']:.3f}")

        save_manifest(
            phase='07_phase2_multilabel_stability',
            metrics={
                'single_split_hamming_loss': {'value': float(hamming), 'fmt': '.4f', 'label': 'Single-split (seed=42) Hamming loss'},
            },
            pipeline_run={
                'dataset_version': 'V6',
                'dataset_path': str(self.input_dir / 'features_engineered.pkl'),
                'script': 'pipeline/00_scripts/prepare_07_phase2_multilabel.py',
            },
            stability=stability,
            meta={'baseline_unweighted_stability': stability_baseline},
        )

        return {
            'model': clf,
            'predictions': y_pred,
            'hamming_loss': hamming,
            'fault_names': self.data['fault_column_names'],
            'data_splits': {
                'X_test': X_test, 'y_test': y_test,
                'idx_train': split['idx_train'], 'idx_test': split['idx_test'],
                'scaler': split['scaler'], 'pca': split['pca'],
                'feature_cols': split['feature_cols'],
            }
        }

    def save_outputs(self, results):
        output_file = self.output_dir / 'multilabel_triggers.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f, protocol=4)
        logger.info(f"\n✓ Saved: {output_file}")

    def run(self):
        logger.info("STEP 07: PHASE 2 - MULTI-LABEL CLASSIFICATION")
        self.load_inputs()
        results = self.process()
        self.save_outputs(results)
        logger.info("STEP 07 COMPLETE!")
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    Step07Processor(args.input, args.output).run()


if __name__ == '__main__':
    main()
