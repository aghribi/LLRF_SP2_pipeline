#!/usr/bin/env python3
"""Test all clustering methods on subset of data.

Tests K-Means, DBSCAN, Hierarchical Clustering, and GMM on up to 100 samples
per fault type to validate implementations before full run.
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

COOKED = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')

def main():
    print("="*70)
    print("CLUSTERING METHODS TEST - SUBSET VALIDATION")
    print("="*70)
    print()

    # Load features
    features_candidates = [
        COOKED / 'step_03_features' / 'features_engineered.pkl',
        COOKED / 'features_engineered.pkl'
    ]

    features_file = None
    for candidate in features_candidates:
        if candidate.exists():
            features_file = candidate
            break

    if features_file is None:
        print(f"ERROR: Features file not found in {COOKED}")
        return

    print(f"Loading features from: {features_file}")
    with open(features_file, 'rb') as f:
        data = pickle.load(f)

    X_pca = data['X_pca']
    y_multilabel = data['y_multilabel']
    fault_names = data['fault_column_names']

    print(f"Total events: {len(X_pca)}")
    print(f"PCA components: {X_pca.shape[1]}")
    print(f"Fault types: {len(fault_names)}")
    print()

    # Test on each fault type with max 100 samples
    for i, fault_name in enumerate(fault_names):
        fault_mask = y_multilabel[:, i] == 1
        X_fault = X_pca[fault_mask]

        # Sample up to 100
        if len(X_fault) > 100:
            np.random.seed(42)
            indices = np.random.choice(len(X_fault), 100, replace=False)
            X_test = X_fault[indices]
        else:
            X_test = X_fault

        if len(X_test) < 5:
            print(f"{fault_name}: Skipped (only {len(X_test)} samples)")
            continue

        print(f"{fault_name} ({len(X_test)} samples):")

        # ===== K-Means =====
        try:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels_km = kmeans.fit_predict(X_test)
            silh_km = silhouette_score(X_test, labels_km)
            unique_km, counts_km = np.unique(labels_km, return_counts=True)
            print(f"  K-Means:      Silhouette={silh_km:.3f}, Clusters={dict(zip(unique_km, counts_km))}")
        except Exception as e:
            print(f"  K-Means:      FAILED - {e}")

        # ===== DBSCAN =====
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels_db = dbscan.fit_predict(X_test)
            n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
            n_outliers_db = list(labels_db).count(-1)
            if n_clusters_db > 1:
                mask_valid = labels_db != -1
                silh_db = silhouette_score(X_test[mask_valid],
                                            labels_db[mask_valid])
                print(f"  DBSCAN:       Silhouette={silh_db:.3f}, "
                      f"Clusters={n_clusters_db}, Outliers={n_outliers_db}")
            else:
                print(f"  DBSCAN:       Only {n_clusters_db} cluster(s) found, "
                      f"Outliers={n_outliers_db}")
        except Exception as e:
            print(f"  DBSCAN:       FAILED - {e}")

        # ===== Hierarchical =====
        try:
            hierarchical = AgglomerativeClustering(n_clusters=3)
            labels_hier = hierarchical.fit_predict(X_test)
            silh_hier = silhouette_score(X_test, labels_hier)
            unique_hier, counts_hier = np.unique(labels_hier, return_counts=True)
            print(f"  Hierarchical: Silhouette={silh_hier:.3f}, Clusters={dict(zip(unique_hier, counts_hier))}")
        except Exception as e:
            print(f"  Hierarchical: FAILED - {e}")

        # ===== GMM =====
        try:
            gmm = GaussianMixture(n_components=3, random_state=42)
            labels_gmm = gmm.fit_predict(X_test)
            silh_gmm = silhouette_score(X_test, labels_gmm)
            unique_gmm, counts_gmm = np.unique(labels_gmm, return_counts=True)
            print(f"  GMM:          Silhouette={silh_gmm:.3f}, Clusters={dict(zip(unique_gmm, counts_gmm))}")
        except Exception as e:
            print(f"  GMM:          FAILED - {e}")

        print()

    print("="*70)
    print("TEST COMPLETE - All methods validated on subset data")
    print("="*70)


if __name__ == '__main__':
    main()
