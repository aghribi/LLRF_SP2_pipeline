#!/usr/bin/env python3
"""
Step 05: Phase 0 - Precursor Detection

Description: Early warning system to predict faults BEFORE they occur using
             precursor features and pre-trigger signal analysis.

Input:  cooked_data/step_03_features/features_engineered.pkl
Output: cooked_data/step_05_phase0/precursor_detection.pkl

Methods:
- Precursor feature-based classification (Random Forest on temporal features)
- Unsupervised anomaly scoring on pre-trigger windows: Isolation Forest, LOF,
  Mahalanobis distance, PCA reconstruction error, DBSCAN
- Ensemble combining all of the above with the RF probability

2026-09-05: added Isolation Forest/LOF/Mahalanobis/PCA-reconstruction/DBSCAN/
Ensemble -- these were named in the pipeline description (see
pipeline/diagrams/) but not implemented; only Random Forest existed before.
Fit on the same leakage-safe X_train (mixed classes, matching this
codebase's existing convention in prepare_04a_dl_anomaly.py rather than a
stricter normal-only novelty-detection split), scored on X_test. DBSCAN has
no train/score-new-points capability in sklearn (transductive), so it's
fit+scored directly on X_test only, consistent with how it's already used
in prepare_04_anomaly_baseline.py and prepare_09_phase3b_clustering.py.

2026-09-06: restricted the feature matrix to leakage_safe_features.
true_precursor_columns() (see load_inputs()) instead of the full feature_cols.
Investigation found the supervised RF's near-1.0 AUC on the full 766-column
matrix was NOT a data artifact (confirmed genuine on the properly-scoped
382-column subset: RF still 0.9998, unsupervised methods on the same columns
still only ~0.65) but the full matrix mixes in whole-signal and unbounded
pre/post statistics that are the wrong kind of feature for this task and were
never intended for it -- see PHASE3_WEAKNESS_DIAGNOSIS_AND_OPERATIONAL_USABILITY.md
section 2.3. Numbers from this script should be described as "early
fault-onset detection ahead of the interlock trip," not "precursor
detection" in the sense of before any physical change begins -- the fixed
window sits immediately before the informatic trigger, which this codebase's
own AMPT-refinement work shows can lag true physical onset.

Usage:
    python prepare_05_phase0_precursor.py \
        --input /path/to/step_03_features \
        --output /path/to/step_05_phase0
"""

import sys
import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA as SkPCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utilities.reporting.manifest import save_manifest

sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import leakage_safe_split, raw_feature_matrix, true_precursor_columns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step_05_phase0.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Step05Processor:
    """Processor for Step 05: Precursor Detection"""

    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_data = None

    def load_inputs(self):
        """Load feature data from Step 03"""
        logger.info("Loading features from Step 03...")

        features_file = self.input_dir / 'features_engineered.pkl'
        with open(features_file, 'rb') as f:
            self.feature_data = pickle.load(f)

        logger.info(f"Loaded feature data:")
        logger.info(f"  Raw features: {self.feature_data['features_all'].shape}, "
                    f"{len(self.feature_data['feature_cols'])} filtered columns")
        if self.feature_data.get('sequences_pretrigger') is not None:
            logger.info(f"  Pre-trigger sequences: {self.feature_data['sequences_pretrigger'].shape}")
        else:
            logger.info(f"  Pre-trigger sequences: Not loaded (using features only)")

        # 2026-09-06: restrict to the fixed-window "true precursor" feature
        # subset (engineer_features() sections 4/6) instead of the full
        # feature_cols matrix. The full matrix mixes in Section 1's
        # whole-signal statistics (no pre/post split at all) and Section 2's
        # unbounded "steady-state" pre/post features (confirmed confounded by
        # a systematic pre-trigger-window-length difference between Normal
        # and Fault event captures) -- neither is appropriate for a "predict
        # before this event's own outcome is known" task. See
        # leakage_safe_features.true_precursor_columns() and
        # PHASE3_WEAKNESS_DIAGNOSIS_AND_OPERATIONAL_USABILITY.md section 2.3.
        all_cols = self.feature_data['feature_cols']
        precursor_cols = true_precursor_columns(all_cols)
        logger.info(f"  Restricting to {len(precursor_cols)} of {len(all_cols)} genuinely "
                    f"pre-trigger, fixed-window 'true precursor' columns (excludes whole-signal "
                    f"and unbounded steady-state features -- see module docstring / Phase 3 doc)")
        self.feature_data['feature_cols'] = precursor_cols

    def process(self):
        """Train precursor detection models"""
        logger.info("="*70)
        logger.info("PHASE 0: PRECURSOR DETECTION")
        logger.info("="*70)

        y = self.feature_data['y_binary']

        if y is None:
            logger.warning("No binary labels available - skipping supervised learning")
            return {}

        y_valid = y

        logger.info(f"Training data: {len(y_valid)} events")
        logger.info(f"  Normal: {np.sum(y_valid == 0)}")
        logger.info(f"  Fault: {np.sum(y_valid == 1)}")

        # Leakage-safe split: scaler fit on the TRAIN fold only.
        split = leakage_safe_split(self.feature_data, y_valid, test_size=0.3, random_state=42)
        X_train, X_test = split['X_train'], split['X_test']
        y_train, y_test = split['y_train'], split['y_test']

        results = {}

        # ========================================
        # Random Forest for Precursor Detection
        # ========================================
        logger.info("\nTraining Random Forest for precursor detection...")

        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)[:, 1]

        logger.info("Random Forest Results:")
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Fault'])}")

        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_proba)
            logger.info(f"ROC AUC: {roc_auc:.3f}")
        else:
            roc_auc = None

        # ========================================
        # Isolation Forest
        # ========================================
        logger.info("\nTraining Isolation Forest for precursor detection...")
        iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
        iso_forest.fit(X_train)
        iso_scores = -iso_forest.decision_function(X_test)  # higher = more anomalous
        iso_preds = (iso_forest.predict(X_test) == -1).astype(int)
        iso_auc = roc_auc_score(y_test, iso_scores) if len(np.unique(y_test)) == 2 else None
        logger.info(f"Isolation Forest ROC AUC: {iso_auc:.3f}" if iso_auc is not None else "Isolation Forest: single-class test set, AUC skipped")
        results['isolation_forest'] = {'model': iso_forest, 'scores': iso_scores, 'predictions': iso_preds, 'roc_auc': iso_auc}

        # ========================================
        # Local Outlier Factor (novelty mode: fit on train, score test)
        # ========================================
        logger.info("\nTraining LOF for precursor detection...")
        lof = LocalOutlierFactor(n_neighbors=20, contamination='auto', novelty=True, n_jobs=-1)
        lof.fit(X_train)
        lof_scores = -lof.decision_function(X_test)  # higher = more anomalous
        lof_preds = (lof.predict(X_test) == -1).astype(int)
        lof_auc = roc_auc_score(y_test, lof_scores) if len(np.unique(y_test)) == 2 else None
        logger.info(f"LOF ROC AUC: {lof_auc:.3f}" if lof_auc is not None else "LOF: single-class test set, AUC skipped")
        results['lof'] = {'scores': lof_scores, 'predictions': lof_preds, 'roc_auc': lof_auc}

        # ========================================
        # Mahalanobis distance (on a PCA-reduced space -- 766 raw features
        # vs. ~1500 train rows makes the full covariance matrix singular;
        # reducing dimensionality first keeps LedoitWolf's shrinkage estimate
        # well-posed)
        # ========================================
        logger.info("\nComputing Mahalanobis distance for precursor detection...")
        maha_pca = SkPCA(n_components=min(30, X_train.shape[0] - 1, X_train.shape[1]), random_state=42)
        X_train_maha = maha_pca.fit_transform(X_train)
        X_test_maha = maha_pca.transform(X_test)
        cov_estimator = LedoitWolf().fit(X_train_maha)
        maha_scores = cov_estimator.mahalanobis(X_test_maha)
        maha_threshold = np.percentile(cov_estimator.mahalanobis(X_train_maha), 95)
        maha_preds = (maha_scores > maha_threshold).astype(int)
        maha_auc = roc_auc_score(y_test, maha_scores) if len(np.unique(y_test)) == 2 else None
        logger.info(f"Mahalanobis distance ROC AUC: {maha_auc:.3f}" if maha_auc is not None else "Mahalanobis: single-class test set, AUC skipped")
        results['mahalanobis'] = {'pca': maha_pca, 'cov_estimator': cov_estimator, 'scores': maha_scores,
                                   'predictions': maha_preds, 'roc_auc': maha_auc}

        # ========================================
        # PCA reconstruction error
        # ========================================
        logger.info("\nComputing PCA reconstruction error for precursor detection...")
        recon_pca = SkPCA(n_components=0.95, random_state=42)
        recon_pca.fit(X_train)
        X_test_recon = recon_pca.inverse_transform(recon_pca.transform(X_test))
        recon_scores = np.mean((X_test - X_test_recon) ** 2, axis=1)
        recon_threshold = np.percentile(
            np.mean((X_train - recon_pca.inverse_transform(recon_pca.transform(X_train))) ** 2, axis=1), 95)
        recon_preds = (recon_scores > recon_threshold).astype(int)
        recon_auc = roc_auc_score(y_test, recon_scores) if len(np.unique(y_test)) == 2 else None
        logger.info(f"PCA reconstruction error ROC AUC: {recon_auc:.3f}" if recon_auc is not None else "PCA reconstruction: single-class test set, AUC skipped")
        results['pca_reconstruction'] = {'pca': recon_pca, 'scores': recon_scores, 'predictions': recon_preds, 'roc_auc': recon_auc}

        # ========================================
        # DBSCAN (transductive -- no fit-then-score-new-points capability in
        # sklearn, so fit+predict directly on X_test, matching how DBSCAN is
        # already used elsewhere in this pipeline, e.g. prepare_04_anomaly_baseline.py)
        # ========================================
        logger.info("\nRunning DBSCAN for precursor detection...")
        dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
        dbscan_labels = dbscan.fit_predict(X_test)
        dbscan_preds = (dbscan_labels == -1).astype(int)
        n_dbscan_outliers = int(np.sum(dbscan_preds))
        logger.info(f"DBSCAN flagged {n_dbscan_outliers}/{len(X_test)} test events as outliers")
        results['dbscan'] = {'model': dbscan, 'cluster_labels': dbscan_labels, 'predictions': dbscan_preds}

        # ========================================
        # Ensemble: min-max normalized average of every anomaly score above
        # (Isolation Forest, LOF, Mahalanobis, PCA reconstruction) plus the
        # supervised RF fault probability. DBSCAN excluded (binary flag, not
        # a continuous score, and evaluated on a different, transductive fit).
        # ========================================
        logger.info("\nComputing ensemble precursor score...")
        def _minmax(a):
            a = np.asarray(a, dtype=float)
            lo, hi = a.min(), a.max()
            return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

        ensemble_score = np.mean([
            _minmax(iso_scores), _minmax(lof_scores), _minmax(maha_scores),
            _minmax(recon_scores), _minmax(y_proba),
        ], axis=0)
        ensemble_auc = roc_auc_score(y_test, ensemble_score) if len(np.unique(y_test)) == 2 else None
        logger.info(f"Ensemble ROC AUC: {ensemble_auc:.3f}" if ensemble_auc is not None else "Ensemble: single-class test set, AUC skipped")
        results['ensemble'] = {'scores': ensemble_score, 'roc_auc': ensemble_auc,
                                'components': ['isolation_forest', 'lof', 'mahalanobis', 'pca_reconstruction', 'random_forest']}

        # Cross-validation: scaler must be refit inside each fold (Pipeline),
        # not once on the globally pre-fit X_valid -- otherwise every fold's
        # "held-out" portion still leaked into the shared scaling statistics.
        X_raw, _ = raw_feature_matrix(self.feature_data)
        cv_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ])
        cv_scores = cross_val_score(cv_pipeline, X_raw, y_valid, cv=5, scoring='f1')
        logger.info(f"Cross-validation F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_data['feature_cols'],
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"\nTop 10 precursor features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        results['random_forest'] = {
            'model': rf_model,
            'predictions': y_pred,
            'probabilities': y_proba,
            'roc_auc': roc_auc,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'test_indices': split['idx_test']
        }

        # Save splits, plus the train-fold-fit scaler/PCA (needed by any
        # downstream script that must transform new rows the same way).
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

        results['metadata'] = {
            'n_events': len(y_valid),
            'n_features': X_train.shape[1],
            'n_faults': np.sum(y_valid == 1),
            'n_normal': np.sum(y_valid == 0)
        }

        # Canonical results manifest -- 2026-09-06, after restricting this
        # script to leakage_safe_features.true_precursor_columns() (see
        # PHASE3_WEAKNESS_DIAGNOSIS_AND_OPERATIONAL_USABILITY.md section 2.3).
        # These numbers should be cited as "early fault-onset detection ahead
        # of the interlock trip," not "precursor detection before any
        # physical change" -- the fixed window sits immediately before the
        # informatic trigger, which can lag true physical onset.
        save_manifest(
            phase='05_phase0_precursor',
            metrics={
                'random_forest_roc_auc': {'value': float(roc_auc), 'fmt': '.4f',
                                           'label': 'Supervised RF ROC AUC (true-precursor feature subset)'},
                'random_forest_cv_f1_mean': {'value': float(cv_scores.mean()), 'fmt': '.3f',
                                              'label': 'RF 5-fold CV F1 (mean)'},
                'random_forest_cv_f1_std': {'value': float(cv_scores.std()), 'fmt': '.3f',
                                             'label': 'RF 5-fold CV F1 (std)'},
                'isolation_forest_roc_auc': {'value': float(results['isolation_forest']['roc_auc']), 'fmt': '.4f',
                                              'label': 'Isolation Forest ROC AUC (unsupervised)'},
                'lof_roc_auc': {'value': float(results['lof']['roc_auc']), 'fmt': '.4f',
                                'label': 'LOF ROC AUC (unsupervised)'},
                'mahalanobis_roc_auc': {'value': float(results['mahalanobis']['roc_auc']), 'fmt': '.4f',
                                         'label': 'Mahalanobis distance ROC AUC (unsupervised)'},
                'pca_reconstruction_roc_auc': {'value': float(results['pca_reconstruction']['roc_auc']), 'fmt': '.4f',
                                                'label': 'PCA reconstruction error ROC AUC (unsupervised)'},
                'ensemble_roc_auc': {'value': float(results['ensemble']['roc_auc']), 'fmt': '.4f',
                                      'label': 'Ensemble ROC AUC (unsupervised methods + RF probability)'},
                'dbscan_outlier_fraction': {'value': float(n_dbscan_outliers) / len(X_test), 'fmt': '.3f',
                                             'label': 'DBSCAN: fraction of test events flagged as outliers'},
                'n_events': {'value': int(len(y_valid)), 'fmt': None, 'label': 'Total events used'},
                'n_features': {'value': int(X_train.shape[1]), 'fmt': None,
                                'label': 'True-precursor feature columns (fixed pre-trigger window)'},
                'n_test': {'value': int(len(X_test)), 'fmt': None, 'label': 'Test-set events'},
            },
            pipeline_run={
                'dataset_version': 'V6',
                'dataset_path': str(self.input_dir / 'features_engineered.pkl'),
                'script': 'pipeline/00_scripts/prepare_05_phase0_precursor.py',
            },
            meta={
                'unsupervised_vs_supervised_gap_note': (
                    "The unsupervised methods (Isolation Forest/LOF/Mahalanobis/PCA-reconstruction) "
                    "never see labels and score 0.58-0.79; the supervised RF and the Ensemble (which "
                    "includes RF's probability) score much higher because there is a real, learnable "
                    "decision boundary in this fixed pre-trigger window -- confirmed genuine, not a "
                    "window-length artifact, via analysis_results_manifest/"
                    "05_precursor_window_leak_investigation.yaml."
                ),
            },
        )

        return results

    def save_outputs(self, results):
        """Save precursor detection results"""
        logger.info("\nSaving results...")

        output_file = self.output_dir / 'precursor_detection.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f, protocol=4)

        file_size_mb = output_file.stat().st_size / (1024**2)

        logger.info("="*70)
        logger.info("SAVED PRECURSOR DETECTION RESULTS")
        logger.info("="*70)
        logger.info(f"File: {output_file}")
        logger.info(f"Size: {file_size_mb:.2f} MB")
        logger.info("")
        logger.info("✓ Precursor detection model ready for early warning")

    def run(self):
        """Execute full pipeline"""
        logger.info("="*70)
        logger.info("STEP 05: PHASE 0 - PRECURSOR DETECTION")
        logger.info("="*70)

        self.load_inputs()
        results = self.process()
        self.save_outputs(results)

        logger.info("")
        logger.info("="*70)
        logger.info("STEP 05 COMPLETE!")
        logger.info("="*70)

        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Step 05: Phase 0 - Precursor Detection')
    parser.add_argument('--input', type=str, required=True, help='Input directory (step_03_features)')
    parser.add_argument('--output', type=str, required=True, help='Output directory (step_05_phase0)')
    args = parser.parse_args()

    processor = Step05Processor(args.input, args.output)
    processor.run()


if __name__ == '__main__':
    main()
