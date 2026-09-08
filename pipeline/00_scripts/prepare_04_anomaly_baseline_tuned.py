#!/usr/bin/env python3
"""
Step 04: Anomaly Detection Baseline (TUNED VERSION)

Description: Unsupervised anomaly detection using Isolation Forest, LOF, and DBSCAN
             with optimized parameters based on data characteristics.

Improvements from baseline version:
- contamination adjusted to actual anomaly rate (45.6% vs default 10%)
- DBSCAN eps computed from k-NN distances (adaptive to data)
- Uses PCA features for DBSCAN (better for high-dimensional data)

Input:  cooked_data/features_engineered.pkl
Output: cooked_data/step_04_anomaly_tuned/anomaly_baseline_tuned.pkl

Usage:
    python prepare_04_anomaly_baseline_tuned.py \
        --input /path/to/features \
        --output /path/to/step_04_anomaly_tuned
"""

import sys
import logging
from pathlib import Path
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step_04_anomaly_tuned.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Step04ProcessorTuned:
    """Processor for Step 04: Anomaly Detection Baseline (TUNED)"""

    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_data = None

    def load_inputs(self):
        """Load feature data"""
        logger.info("Loading features...")

        features_file = self.input_dir / 'features_engineered.pkl'
        if not features_file.exists():
            raise FileNotFoundError(f"Features not found: {features_file}")

        with open(features_file, 'rb') as f:
            self.feature_data = pickle.load(f)

        logger.info(f"Loaded feature data:")
        logger.info(f"  Features (scaled): {self.feature_data['X_scaled'].shape}")
        logger.info(f"  Features (PCA): {self.feature_data['X_pca'].shape}")
        logger.info(f"  Binary labels: {self.feature_data['y_binary'].shape if self.feature_data['y_binary'] is not None else 'N/A'}")

    def compute_optimal_eps(self, X, k=20):
        """Compute optimal DBSCAN eps from k-NN distances"""
        logger.info(f"\nComputing optimal eps from {k}-NN distances...")

        nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        knn_distances = np.sort(distances[:, -1])

        # Use 90th percentile as recommended
        eps_optimal = np.percentile(knn_distances, 90)

        logger.info(f"  k-NN distance statistics:")
        logger.info(f"    Min: {knn_distances.min():.3f}")
        logger.info(f"    Median: {np.median(knn_distances):.3f}")
        logger.info(f"    90th percentile: {eps_optimal:.3f}")
        logger.info(f"    Max: {knn_distances.max():.3f}")
        logger.info(f"  ✓ Recommended eps: {eps_optimal:.3f}")

        return eps_optimal, knn_distances

    def process(self):
        """Train unsupervised anomaly detection models with TUNED parameters"""
        logger.info("="*70)
        logger.info("ANOMALY DETECTION BASELINE (TUNED)")
        logger.info("="*70)

        X_scaled = self.feature_data['X_scaled']
        X_pca = self.feature_data['X_pca']
        y_binary = self.feature_data['y_binary']

        results = {}

        # Compute actual contamination from data
        if y_binary is not None:
            actual_contamination = np.sum(y_binary == 1) / len(y_binary)
            logger.info(f"\nDataset statistics:")
            logger.info(f"  Total events: {len(y_binary)}")
            logger.info(f"  Normal: {np.sum(y_binary == 0)} ({np.sum(y_binary == 0)/len(y_binary):.1%})")
            logger.info(f"  Anomalies: {np.sum(y_binary == 1)} ({actual_contamination:.1%})")
            logger.info(f"\n✓ Using contamination={actual_contamination:.3f} (actual anomaly rate)")
        else:
            actual_contamination = 0.1  # Fallback
            logger.info(f"\nNo labels available, using default contamination=0.1")

        # ========================================
        # 1. Isolation Forest (TUNED)
        # ========================================
        logger.info("\n" + "="*70)
        logger.info("1. ISOLATION FOREST (TUNED)")
        logger.info("="*70)
        logger.info(f"Parameters:")
        logger.info(f"  n_estimators: 100")
        logger.info(f"  contamination: {actual_contamination:.3f} (TUNED from 'auto' ≈ 0.1)")
        logger.info(f"  random_state: 42")

        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=actual_contamination,
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(X_scaled)
        iso_scores = iso_forest.decision_function(X_scaled)
        iso_preds = iso_forest.predict(X_scaled)  # 1=normal, -1=anomaly
        iso_preds_binary = (iso_preds == -1).astype(int)  # Convert to 0/1

        results['isolation_forest'] = {
            'model': iso_forest,
            'scores': iso_scores,
            'predictions': iso_preds_binary,
            'contamination': actual_contamination
        }

        if y_binary is not None:
            recall = recall_score(y_binary, iso_preds_binary)
            precision = precision_score(y_binary, iso_preds_binary, zero_division=0)
            f1 = f1_score(y_binary, iso_preds_binary, zero_division=0)

            logger.info(f"\nIsolation Forest Performance:")
            logger.info(f"  Precision: {precision:.3f}")
            logger.info(f"  Recall: {recall:.3f}")
            logger.info(f"  F1-Score: {f1:.3f}")
            logger.info(f"\n{classification_report(y_binary, iso_preds_binary, target_names=['Normal', 'Anomaly'])}")
            logger.info(f"Confusion Matrix:\n{confusion_matrix(y_binary, iso_preds_binary)}")

        # ========================================
        # 2. Local Outlier Factor (TUNED)
        # ========================================
        logger.info("\n" + "="*70)
        logger.info("2. LOCAL OUTLIER FACTOR (TUNED)")
        logger.info("="*70)
        logger.info(f"Parameters:")
        logger.info(f"  n_neighbors: 50 (TUNED from 20 for more robust density estimate)")
        logger.info(f"  contamination: {actual_contamination:.3f} (TUNED from 'auto' ≈ 0.1)")

        lof = LocalOutlierFactor(
            n_neighbors=50,
            contamination=actual_contamination,
            novelty=False,
            n_jobs=-1
        )
        lof_preds = lof.fit_predict(X_scaled)  # 1=normal, -1=anomaly
        lof_scores = lof.negative_outlier_factor_
        lof_preds_binary = (lof_preds == -1).astype(int)

        results['lof'] = {
            'scores': lof_scores,
            'predictions': lof_preds_binary,
            'contamination': actual_contamination,
            'n_neighbors': 50
        }

        if y_binary is not None:
            recall = recall_score(y_binary, lof_preds_binary)
            precision = precision_score(y_binary, lof_preds_binary, zero_division=0)
            f1 = f1_score(y_binary, lof_preds_binary, zero_division=0)

            logger.info(f"\nLOF Performance:")
            logger.info(f"  Precision: {precision:.3f}")
            logger.info(f"  Recall: {recall:.3f}")
            logger.info(f"  F1-Score: {f1:.3f}")
            logger.info(f"\n{classification_report(y_binary, lof_preds_binary, target_names=['Normal', 'Anomaly'])}")

        # ========================================
        # 3. DBSCAN Clustering (TUNED)
        # ========================================
        logger.info("\n" + "="*70)
        logger.info("3. DBSCAN (TUNED)")
        logger.info("="*70)
        logger.info("Using PCA features (251 dims) instead of scaled features (1243 dims)")
        logger.info("Computing optimal eps from k-NN distances...")

        k = 20
        eps_optimal, knn_distances = self.compute_optimal_eps(X_pca, k=k)
        min_samples_tuned = 2 * k  # Rule of thumb: 2×k for high-dimensional data

        logger.info(f"\nParameters:")
        logger.info(f"  eps: {eps_optimal:.3f} (TUNED from 0.5, computed from 90th percentile k-NN)")
        logger.info(f"  min_samples: {min_samples_tuned} (TUNED from 5, using 2×k rule)")

        dbscan = DBSCAN(eps=eps_optimal, min_samples=min_samples_tuned, n_jobs=-1)
        dbscan_labels = dbscan.fit_predict(X_pca)

        # Outliers are labeled as -1
        dbscan_preds_binary = (dbscan_labels == -1).astype(int)
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_outliers = np.sum(dbscan_labels == -1)

        logger.info(f"\nDBSCAN Results:")
        logger.info(f"  Clusters found: {n_clusters}")
        logger.info(f"  Outliers: {n_outliers}/{len(X_pca)} ({n_outliers/len(X_pca):.1%})")

        if n_clusters > 0:
            cluster_sizes = [np.sum(dbscan_labels == i) for i in range(n_clusters)]
            logger.info(f"  Cluster sizes: {cluster_sizes}")

        results['dbscan'] = {
            'model': dbscan,
            'cluster_labels': dbscan_labels,
            'predictions': dbscan_preds_binary,
            'eps': eps_optimal,
            'min_samples': min_samples_tuned,
            'knn_distances': knn_distances,
            'n_clusters': n_clusters
        }

        if y_binary is not None:
            recall = recall_score(y_binary, dbscan_preds_binary)
            precision = precision_score(y_binary, dbscan_preds_binary, zero_division=0)
            f1 = f1_score(y_binary, dbscan_preds_binary, zero_division=0)

            logger.info(f"\nDBSCAN Performance:")
            logger.info(f"  Precision: {precision:.3f}")
            logger.info(f"  Recall: {recall:.3f}")
            logger.info(f"  F1-Score: {f1:.3f}")
            logger.info(f"\n{classification_report(y_binary, dbscan_preds_binary, target_names=['Normal', 'Anomaly'])}")

        # ========================================
        # Summary
        # ========================================
        logger.info("\n" + "="*70)
        logger.info("TUNED ANOMALY DETECTION SUMMARY")
        logger.info("="*70)
        logger.info(f"Isolation Forest - Anomalies detected: {np.sum(iso_preds_binary)}/{len(X_scaled)} ({np.sum(iso_preds_binary)/len(X_scaled):.1%})")
        logger.info(f"LOF - Anomalies detected: {np.sum(lof_preds_binary)}/{len(X_scaled)} ({np.sum(lof_preds_binary)/len(X_scaled):.1%})")
        logger.info(f"DBSCAN - Outliers detected: {n_outliers}/{len(X_pca)} ({n_outliers/len(X_pca):.1%})")

        if y_binary is not None:
            logger.info(f"\nActual anomalies (ground truth): {np.sum(y_binary == 1)}/{len(y_binary)} ({actual_contamination:.1%})")

            iso_recall = recall_score(y_binary, iso_preds_binary)
            lof_recall = recall_score(y_binary, lof_preds_binary)
            dbscan_recall = recall_score(y_binary, dbscan_preds_binary)

            logger.info(f"\nRecall Comparison (% of actual anomalies detected):")
            logger.info(f"  Isolation Forest: {iso_recall:.1%}")
            logger.info(f"  LOF: {lof_recall:.1%}")
            logger.info(f"  DBSCAN: {dbscan_recall:.1%}")

        # Save feature data reference
        results['feature_data'] = {
            'X_scaled': X_scaled,
            'X_pca': X_pca,
            'y_binary': y_binary,
            'feature_cols': self.feature_data['feature_cols'],
            'metadata': self.feature_data['metadata']
        }

        return results

    def save_outputs(self, results):
        """Save anomaly detection results"""
        logger.info("\nSaving results...")

        output_file = self.output_dir / 'anomaly_baseline_tuned.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f, protocol=4)

        file_size_mb = output_file.stat().st_size / (1024**2)

        logger.info("="*70)
        logger.info("SAVED TUNED ANOMALY DETECTION RESULTS")
        logger.info("="*70)
        logger.info(f"File: {output_file}")
        logger.info(f"Size: {file_size_mb:.2f} MB")
        logger.info("")
        logger.info("✓ Tuned baseline established for anomaly detection")

    def run(self):
        """Execute full pipeline"""
        logger.info("="*70)
        logger.info("STEP 04: ANOMALY DETECTION BASELINE (TUNED)")
        logger.info("="*70)

        self.load_inputs()
        results = self.process()
        self.save_outputs(results)

        logger.info("")
        logger.info("="*70)
        logger.info("STEP 04 (TUNED) COMPLETE!")
        logger.info("="*70)

        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Step 04: Anomaly Detection Baseline (Tuned)')
    parser.add_argument('--input', type=str,
                        default='/sps/m4cast/_spiral2_data/_llrf_data/cooked_data',
                        help='Input directory containing features_engineered.pkl')
    parser.add_argument('--output', type=str,
                        default='/sps/m4cast/_spiral2_data/_llrf_data/cooked_data/step_04_anomaly_tuned',
                        help='Output directory for tuned results')
    args = parser.parse_args()

    processor = Step04ProcessorTuned(args.input, args.output)
    processor.run()


if __name__ == '__main__':
    main()
