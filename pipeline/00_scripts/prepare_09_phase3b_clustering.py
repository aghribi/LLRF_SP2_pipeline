#!/usr/bin/env python3
"""
Step 09: Phase 3B - Sub-Type Clustering

Description: Discover sub-mechanisms within each fault type using clustering.

Input:  cooked_data/step_03_features/features_engineered.pkl
Output: cooked_data/step_09_phase3b/subtype_clustering.pkl
"""

import sys
import logging
from pathlib import Path
import pickle
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler('step_09_phase3b.log'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


class Step09Processor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_inputs(self):
        with open(self.input_dir / 'features_engineered.pkl', 'rb') as f:
            self.data = pickle.load(f)
        logger.info(f"Loaded: {self.data['X_scaled'].shape}")

    def process(self):
        logger.info("="*70)
        logger.info("PHASE 3B: SUB-TYPE CLUSTERING (MULTI-METHOD)")
        logger.info("="*70)

        X = self.data['X_pca']  # Use PCA for clustering
        y_multilabel = self.data['y_multilabel']

        results = {}

        # Cluster per fault type
        for i, fault_name in enumerate(self.data['fault_column_names']):
            fault_mask = y_multilabel[:, i] == 1
            if fault_mask.sum() < 5:
                logger.info(f"\nSkipping {fault_name}: too few samples ({fault_mask.sum()})")
                continue

            X_fault = X[fault_mask]
            logger.info(f"\n{fault_name}: {len(X_fault)} events")

            # Determine n_clusters for methods that need it
            n_clusters = min(3, len(X_fault) // 2)  # Max 3 clusters
            if n_clusters < 2:
                logger.info(f"  Skipping (too few events for clustering)")
                continue

            # ===== METHOD 1: K-MEANS =====
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels_km = kmeans.fit_predict(X_fault)
            silh_km = silhouette_score(X_fault, labels_km)
            logger.info(f"  K-Means (k={n_clusters}): Silh={silh_km:.3f}")

            unique_km, counts_km = np.unique(labels_km, return_counts=True)
            for cluster_id, count in zip(unique_km, counts_km):
                logger.info(f"    Cluster {cluster_id}: {count} events")

            # ===== METHOD 2: DBSCAN =====
            # Auto-tune eps using k-distance plot heuristic
            neighbors = NearestNeighbors(n_neighbors=5)
            neighbors.fit(X_fault)
            distances, _ = neighbors.kneighbors(X_fault)
            eps = np.percentile(distances[:, -1], 75)  # 75th percentile

            dbscan = DBSCAN(eps=eps, min_samples=5)
            labels_db = dbscan.fit_predict(X_fault)
            n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
            n_outliers_db = list(labels_db).count(-1)

            if n_clusters_db > 1:
                mask_valid = labels_db != -1
                silh_db = silhouette_score(X_fault[mask_valid], labels_db[mask_valid])
                logger.info(f"  DBSCAN (eps={eps:.3f}): Silh={silh_db:.3f}, "
                           f"Clusters={n_clusters_db}, Outliers={n_outliers_db}")
            else:
                silh_db = 0.0
                logger.info(f"  DBSCAN: Only {n_clusters_db} cluster(s), Outliers={n_outliers_db}")

            # ===== METHOD 3: HIERARCHICAL =====
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            labels_hier = hierarchical.fit_predict(X_fault)
            silh_hier = silhouette_score(X_fault, labels_hier)
            logger.info(f"  Hierarchical (k={n_clusters}): Silh={silh_hier:.3f}")

            # ===== METHOD 4: GMM =====
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            labels_gmm = gmm.fit_predict(X_fault)
            proba_gmm = gmm.predict_proba(X_fault)
            silh_gmm = silhouette_score(X_fault, labels_gmm)
            logger.info(f"  GMM (k={n_clusters}): Silh={silh_gmm:.3f}")

            # Store all results
            results[fault_name] = {
                'kmeans': {
                    'model': kmeans,
                    'labels': labels_km,
                    'silhouette': silh_km,
                    'n_clusters': n_clusters
                },
                'dbscan': {
                    'model': dbscan,
                    'labels': labels_db,
                    'silhouette': silh_db,
                    'n_clusters': n_clusters_db,
                    'n_outliers': n_outliers_db,
                    'eps': eps
                },
                'hierarchical': {
                    'model': hierarchical,
                    'labels': labels_hier,
                    'silhouette': silh_hier,
                    'n_clusters': n_clusters
                },
                'gmm': {
                    'model': gmm,
                    'labels': labels_gmm,
                    'probabilities': proba_gmm,
                    'silhouette': silh_gmm,
                    'n_clusters': n_clusters
                }
            }

        return results

    def save_outputs(self, results):
        output_file = self.output_dir / 'subtype_clustering.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f, protocol=4)
        logger.info(f"\n✓ Saved: {output_file}")

    def run(self):
        logger.info("STEP 09: PHASE 3B - SUB-TYPE CLUSTERING")
        self.load_inputs()
        results = self.process()
        self.save_outputs(results)
        logger.info("STEP 09 COMPLETE!")
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    Step09Processor(args.input, args.output).run()


if __name__ == '__main__':
    main()
