#!/usr/bin/env python3
"""
Step 04: Anomaly Detection Baseline

Description: Unsupervised anomaly detection using Isolation Forest, LOF, and DBSCAN.
             Establishes baseline for normal vs anomalous behavior without labels.

Input:  cooked_data/step_03_features/features_engineered.pkl
Output: cooked_data/step_04_anomaly/anomaly_baseline.pkl

Models:
- Isolation Forest: Outlier detection via random partitioning
- Local Outlier Factor (LOF): Density-based anomaly scoring
- DBSCAN: Clustering to identify outlier points

Usage:
    python prepare_04_anomaly_baseline.py \
        --input /path/to/step_03_features \
        --output /path/to/step_04_anomaly
"""

import sys
import logging
from pathlib import Path
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step_04_anomaly.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Step04Processor:
    """Processor for Step 04: Anomaly Detection Baseline"""

    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_data = None

    def load_inputs(self):
        """Load feature data from Step 03"""
        logger.info("Loading features from Step 03...")

        features_file = self.input_dir / 'features_engineered.pkl'
        if not features_file.exists():
            raise FileNotFoundError(f"Features not found: {features_file}")

        with open(features_file, 'rb') as f:
            self.feature_data = pickle.load(f)

        logger.info(f"Loaded feature data:")
        logger.info(f"  Features (scaled): {self.feature_data['X_scaled'].shape}")
        logger.info(f"  Features (PCA): {self.feature_data['X_pca'].shape}")
        logger.info(f"  Binary labels: {self.feature_data['y_binary'].shape if self.feature_data['y_binary'] is not None else 'N/A'}")

    def process(self):
        """Train unsupervised anomaly detection models"""
        logger.info("="*70)
        logger.info("ANOMALY DETECTION BASELINE")
        logger.info("="*70)

        X = self.feature_data['X_scaled']
        y_binary = self.feature_data['y_binary']

        results = {}

        # ========================================
        # 1. Isolation Forest
        # ========================================
        logger.info("\nTraining Isolation Forest...")
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination='auto',
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(X)
        iso_scores = iso_forest.decision_function(X)
        iso_preds = iso_forest.predict(X)  # 1=normal, -1=anomaly
        iso_preds_binary = (iso_preds == -1).astype(int)  # Convert to 0/1

        results['isolation_forest'] = {
            'model': iso_forest,
            'scores': iso_scores,
            'predictions': iso_preds_binary
        }

        if y_binary is not None:
            logger.info("Isolation Forest vs Ground Truth:")
            logger.info(f"\n{classification_report(y_binary, iso_preds_binary, target_names=['Normal', 'Anomaly'])}")
            logger.info(f"Confusion Matrix:\n{confusion_matrix(y_binary, iso_preds_binary)}")

        # ========================================
        # 2. Local Outlier Factor
        # ========================================
        logger.info("\nTraining Local Outlier Factor...")
        lof = LocalOutlierFactor(n_neighbors=20, contamination='auto', novelty=False, n_jobs=-1)
        lof_preds = lof.fit_predict(X)  # 1=normal, -1=anomaly
        lof_scores = lof.negative_outlier_factor_
        lof_preds_binary = (lof_preds == -1).astype(int)

        results['lof'] = {
            'scores': lof_scores,
            'predictions': lof_preds_binary
        }

        if y_binary is not None:
            logger.info("LOF vs Ground Truth:")
            logger.info(f"\n{classification_report(y_binary, lof_preds_binary, target_names=['Normal', 'Anomaly'])}")

        # ========================================
        # 3. DBSCAN Clustering
        # ========================================
        logger.info("\nTraining DBSCAN...")
        dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
        dbscan_labels = dbscan.fit_predict(X)

        # Outliers are labeled as -1
        dbscan_preds_binary = (dbscan_labels == -1).astype(int)
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_outliers = np.sum(dbscan_labels == -1)

        logger.info(f"DBSCAN found {n_clusters} clusters and {n_outliers} outliers")

        results['dbscan'] = {
            'model': dbscan,
            'cluster_labels': dbscan_labels,
            'predictions': dbscan_preds_binary
        }

        if y_binary is not None:
            logger.info("DBSCAN vs Ground Truth:")
            logger.info(f"\n{classification_report(y_binary, dbscan_preds_binary, target_names=['Normal', 'Anomaly'])}")

        # Summary
        logger.info("\n" + "="*70)
        logger.info("ANOMALY DETECTION SUMMARY")
        logger.info("="*70)
        logger.info(f"Isolation Forest - Anomalies detected: {np.sum(iso_preds_binary)}/{len(X)}")
        logger.info(f"LOF - Anomalies detected: {np.sum(lof_preds_binary)}/{len(X)}")
        logger.info(f"DBSCAN - Outliers detected: {n_outliers}/{len(X)}")

        # Save feature data reference
        results['feature_data'] = {
            'X_scaled': X,
            'y_binary': y_binary,
            'feature_cols': self.feature_data['feature_cols'],
            'metadata': self.feature_data['metadata']
        }

        return results

    def save_outputs(self, results):
        """Save anomaly detection results"""
        logger.info("\nSaving results...")

        output_file = self.output_dir / 'anomaly_baseline.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f, protocol=4)

        file_size_mb = output_file.stat().st_size / (1024**2)

        logger.info("="*70)
        logger.info("SAVED ANOMALY DETECTION RESULTS")
        logger.info("="*70)
        logger.info(f"File: {output_file}")
        logger.info(f"Size: {file_size_mb:.2f} MB")
        logger.info("")
        logger.info("✓ Baseline established for anomaly detection")

    def run(self):
        """Execute full pipeline"""
        logger.info("="*70)
        logger.info("STEP 04: ANOMALY DETECTION BASELINE")
        logger.info("="*70)

        self.load_inputs()
        results = self.process()
        self.save_outputs(results)

        logger.info("")
        logger.info("="*70)
        logger.info("STEP 04 COMPLETE!")
        logger.info("="*70)

        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Step 04: Anomaly Detection Baseline')
    parser.add_argument('--input', type=str, required=True, help='Input directory (step_03_features)')
    parser.add_argument('--output', type=str, required=True, help='Output directory (step_04_anomaly)')
    args = parser.parse_args()

    processor = Step04Processor(args.input, args.output)
    processor.run()


if __name__ == '__main__':
    main()
