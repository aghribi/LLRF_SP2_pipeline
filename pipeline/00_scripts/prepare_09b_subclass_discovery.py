#!/usr/bin/env python3
"""
Step 09b: Enhanced Subclass Discovery
=====================================
Discovers fault subtypes targeting ~19 known subclasses from expert knowledge.

Known Subclass Hierarchy:
1. Courant Pick up
2. Circulator Arc
3. RF PLC NOK (SAF, Cryo, Ampli, UGSx, E/T Circu)
4. Coupler Vacuum NOK (Leak/outgassing, Ipu)
5. Cavity BD or Quench (Sudden Wcav dissipation)
6. Sudden Wcav diss. False
7. RF Protections (Pr max, Ucav max)
8. Ecav Instability (Loop oscillation, Taconis, Power chain gain, Quench)
9. Quench aside (Beam loss)

Methods:
- Hierarchical clustering with dendrogram analysis
- HDBSCAN for density-based discovery
- Gaussian Mixture Models for probabilistic assignment
- Spectral clustering for non-convex clusters
- Ensemble consensus clustering

Input:  cooked_data/features_engineered.pkl
Output: cooked_data/step_09b_subclass_discovery/
"""

import sys
import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# Try to import HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("HDBSCAN not available, using DBSCAN instead")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step_09b_subclass.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Known subclass structure (target: ~19 subclasses)
KNOWN_SUBCLASSES = {
    'Courant Pick up': [],
    'Circulator Arc': [],
    'RF PLC NOK': ['SAF', 'Cryo', 'Ampli', 'UGSx', 'E/T Circu'],
    'Coupler Vacuum NOK': ['Leak/outgassing', 'Ipu'],
    'Cavity BD or Quench': ['Sudden Wcav dissipation'],
    'Sudden Wcav diss. False': [],
    'RF Protections': ['Pr max', 'Ucav max'],
    'Ecav Instability': ['Loop oscillation', 'Taconis', 'Power chain gain', 'Quench'],
    'Quench aside': ['Beam loss']
}


class SubclassDiscovery:
    def __init__(self, input_path, output_dir):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def load_data(self):
        logger.info("Loading data...")
        with open(self.input_path, 'rb') as f:
            self.data = pickle.load(f)

        self.X = self.data['X_scaled']
        self.X_pca = self.data.get('X_pca', self.X)
        self.y_multilabel = self.data['y_multilabel']
        self.fault_names = self.data.get('fault_column_names',
                                         [f'Fault_{i}' for i in range(self.y_multilabel.shape[1])])

        # Get fault mask (any fault)
        self.fault_mask = self.y_multilabel.sum(axis=1) > 0
        self.X_faults = self.X_pca[self.fault_mask]
        self.y_faults = self.y_multilabel[self.fault_mask]

        logger.info(f"Total events: {len(self.X)}")
        logger.info(f"Fault events: {len(self.X_faults)} ({len(self.X_faults)/len(self.X)*100:.1f}%)")
        logger.info(f"Features: {self.X.shape[1]}, PCA components: {self.X_pca.shape[1]}")

    def find_optimal_k(self, X, k_range=(2, 20)):
        """Find optimal number of clusters using multiple metrics."""
        logger.info(f"Finding optimal k in range {k_range}...")

        metrics = {
            'k': [],
            'silhouette': [],
            'calinski': [],
            'davies_bouldin': [],
            'inertia': []
        }

        for k in range(k_range[0], min(k_range[1] + 1, len(X))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            metrics['k'].append(k)
            metrics['silhouette'].append(silhouette_score(X, labels))
            metrics['calinski'].append(calinski_harabasz_score(X, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(X, labels))
            metrics['inertia'].append(kmeans.inertia_)

        # Find optimal k (maximize silhouette, minimize davies-bouldin)
        df = pd.DataFrame(metrics)

        if len(df) == 1:
            # Only one k value, return it
            optimal_k = df['k'].iloc[0]
            logger.info(f"Only one k value, using k={int(optimal_k)}")
            return int(optimal_k), df

        # Normalize metrics for comparison
        silh_range = df['silhouette'].max() - df['silhouette'].min()
        db_range = df['davies_bouldin'].max() - df['davies_bouldin'].min()

        if silh_range > 0:
            df['silh_norm'] = (df['silhouette'] - df['silhouette'].min()) / silh_range
        else:
            df['silh_norm'] = 0.5

        if db_range > 0:
            df['db_norm'] = 1 - (df['davies_bouldin'] - df['davies_bouldin'].min()) / db_range
        else:
            df['db_norm'] = 0.5

        df['score'] = (df['silh_norm'] + df['db_norm']) / 2

        optimal_k = df.loc[df['score'].idxmax(), 'k']
        logger.info(f"Optimal k: {int(optimal_k)}")

        return int(optimal_k), df

    def cluster_all_faults(self, target_k=19):
        """Cluster all fault events together targeting ~19 subclasses."""
        logger.info("="*70)
        logger.info("CLUSTERING ALL FAULT EVENTS")
        logger.info(f"Target subclasses: {target_k}")
        logger.info("="*70)

        X = self.X_faults
        results = {}

        # Find optimal k
        optimal_k, k_metrics = self.find_optimal_k(X, k_range=(5, 25))
        results['k_metrics'] = k_metrics
        results['optimal_k'] = optimal_k

        # Use target k for main analysis
        k = target_k

        # ===== 1. K-MEANS =====
        logger.info(f"\n1. K-Means (k={k})")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels_km = kmeans.fit_predict(X)
        silh_km = silhouette_score(X, labels_km)
        logger.info(f"   Silhouette: {silh_km:.4f}")
        results['kmeans'] = {
            'labels': labels_km,
            'silhouette': silh_km,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_
        }

        # ===== 2. HIERARCHICAL =====
        logger.info(f"\n2. Hierarchical Clustering (k={k})")
        hier = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels_hier = hier.fit_predict(X)
        silh_hier = silhouette_score(X, labels_hier)
        logger.info(f"   Silhouette: {silh_hier:.4f}")

        # Also compute linkage for dendrogram
        linkage_matrix = linkage(X[:1000], method='ward')  # Sample for speed
        results['hierarchical'] = {
            'labels': labels_hier,
            'silhouette': silh_hier,
            'linkage': linkage_matrix
        }

        # ===== 3. GMM =====
        logger.info(f"\n3. Gaussian Mixture Model (k={k})")
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
        labels_gmm = gmm.fit_predict(X)
        proba_gmm = gmm.predict_proba(X)
        silh_gmm = silhouette_score(X, labels_gmm)
        logger.info(f"   Silhouette: {silh_gmm:.4f}")
        results['gmm'] = {
            'labels': labels_gmm,
            'probabilities': proba_gmm,
            'silhouette': silh_gmm,
            'means': gmm.means_,
            'weights': gmm.weights_
        }

        # ===== 4. SPECTRAL =====
        logger.info(f"\n4. Spectral Clustering (k={k})")
        # Use subset for spectral (memory intensive)
        n_sample = min(2000, len(X))
        idx_sample = np.random.RandomState(42).choice(len(X), n_sample, replace=False)
        X_sample = X[idx_sample]

        spectral = SpectralClustering(n_clusters=k, random_state=42, affinity='nearest_neighbors')
        labels_spec_sample = spectral.fit_predict(X_sample)
        silh_spec = silhouette_score(X_sample, labels_spec_sample)
        logger.info(f"   Silhouette (sample): {silh_spec:.4f}")

        # Assign remaining points to nearest cluster center
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_sample, labels_spec_sample)
        labels_spec = knn.predict(X)

        results['spectral'] = {
            'labels': labels_spec,
            'silhouette': silh_spec,
            'sample_idx': idx_sample
        }

        # ===== 5. HDBSCAN or DBSCAN =====
        if HDBSCAN_AVAILABLE:
            logger.info("\n5. HDBSCAN (density-based)")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5)
            labels_hdb = clusterer.fit_predict(X)
            n_clusters_hdb = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
            n_noise = (labels_hdb == -1).sum()

            if n_clusters_hdb > 1:
                mask_valid = labels_hdb != -1
                silh_hdb = silhouette_score(X[mask_valid], labels_hdb[mask_valid])
            else:
                silh_hdb = 0.0

            logger.info(f"   Clusters: {n_clusters_hdb}, Noise: {n_noise}")
            logger.info(f"   Silhouette: {silh_hdb:.4f}")
            results['hdbscan'] = {
                'labels': labels_hdb,
                'silhouette': silh_hdb,
                'n_clusters': n_clusters_hdb,
                'n_noise': n_noise,
                'probabilities': clusterer.probabilities_
            }
        else:
            logger.info("\n5. DBSCAN (density-based)")
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=10)
            nn.fit(X)
            distances, _ = nn.kneighbors(X)
            eps = np.percentile(distances[:, -1], 90)

            dbscan = DBSCAN(eps=eps, min_samples=10)
            labels_db = dbscan.fit_predict(X)
            n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
            n_noise = (labels_db == -1).sum()

            if n_clusters_db > 1:
                mask_valid = labels_db != -1
                silh_db = silhouette_score(X[mask_valid], labels_db[mask_valid])
            else:
                silh_db = 0.0

            logger.info(f"   Clusters: {n_clusters_db}, Noise: {n_noise}, eps: {eps:.4f}")
            logger.info(f"   Silhouette: {silh_db:.4f}")
            results['dbscan'] = {
                'labels': labels_db,
                'silhouette': silh_db,
                'n_clusters': n_clusters_db,
                'n_noise': n_noise,
                'eps': eps
            }

        # ===== ENSEMBLE CONSENSUS =====
        logger.info("\n6. Ensemble Consensus")
        # Simple voting across methods
        all_labels = np.column_stack([
            labels_km, labels_hier, labels_gmm, labels_spec
        ])

        # Use co-association matrix for consensus
        from sklearn.cluster import AgglomerativeClustering as HC
        n_samples = len(X)
        coassoc = np.zeros((n_samples, n_samples))

        for i in range(all_labels.shape[1]):
            labels = all_labels[:, i]
            for c in np.unique(labels):
                mask = labels == c
                coassoc[np.ix_(mask, mask)] += 1

        coassoc /= all_labels.shape[1]  # Normalize

        # Cluster the co-association matrix
        # Use 'metric' for newer sklearn versions
        try:
            consensus = HC(n_clusters=k, metric='precomputed', linkage='average')
        except TypeError:
            consensus = HC(n_clusters=k, affinity='precomputed', linkage='average')
        labels_consensus = consensus.fit_predict(1 - coassoc)
        silh_consensus = silhouette_score(X, labels_consensus)
        logger.info(f"   Silhouette: {silh_consensus:.4f}")

        results['consensus'] = {
            'labels': labels_consensus,
            'silhouette': silh_consensus,
            'coassociation': coassoc
        }

        # ===== SUMMARY =====
        logger.info("\n" + "="*70)
        logger.info("CLUSTERING SUMMARY")
        logger.info("="*70)
        methods = ['kmeans', 'hierarchical', 'gmm', 'spectral', 'consensus']
        if 'hdbscan' in results:
            methods.append('hdbscan')
        elif 'dbscan' in results:
            methods.append('dbscan')

        for method in methods:
            if method in results:
                silh = results[method]['silhouette']
                logger.info(f"  {method:15}: Silhouette = {silh:.4f}")

        # Best method
        best_method = max(methods, key=lambda m: results.get(m, {}).get('silhouette', 0))
        logger.info(f"\n  Best method: {best_method} (Silh = {results[best_method]['silhouette']:.4f})")
        results['best_method'] = best_method

        return results

    def cluster_per_fault_type(self):
        """Cluster within each fault type to find subtypes."""
        logger.info("\n" + "="*70)
        logger.info("CLUSTERING PER FAULT TYPE")
        logger.info("="*70)

        per_fault_results = {}

        for i, fault_name in enumerate(self.fault_names):
            mask = self.y_multilabel[:, i] == 1
            n_events = mask.sum()

            if n_events < 10:
                logger.info(f"\n{fault_name}: Skipped (only {n_events} events)")
                continue

            X_fault = self.X_pca[mask]
            logger.info(f"\n{fault_name}: {n_events} events")

            # Determine k (2-5 subtypes per fault)
            max_k = min(5, n_events // 10)
            if max_k < 2:
                logger.info("  Too few events for subtype discovery")
                continue

            # Find optimal k
            optimal_k, _ = self.find_optimal_k(X_fault, k_range=(2, max_k))
            k = optimal_k

            # K-Means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_fault)
            silh = silhouette_score(X_fault, labels)

            logger.info(f"  K-Means (k={k}): Silhouette = {silh:.4f}")

            # Cluster sizes
            unique, counts = np.unique(labels, return_counts=True)
            for c, cnt in zip(unique, counts):
                logger.info(f"    Subtype {c}: {cnt} events ({cnt/n_events*100:.1f}%)")

            per_fault_results[fault_name] = {
                'labels': labels,
                'silhouette': silh,
                'k': k,
                'centers': kmeans.cluster_centers_,
                'cluster_sizes': dict(zip(unique.tolist(), counts.tolist()))
            }

        return per_fault_results

    def analyze_cluster_characteristics(self, labels, method_name='best'):
        """Analyze what distinguishes each cluster."""
        logger.info("\n" + "="*70)
        logger.info(f"CLUSTER CHARACTERISTICS ({method_name})")
        logger.info("="*70)

        X = self.X_faults
        feature_names = self.data.get('feature_names',
                                      [f'feat_{i}' for i in range(X.shape[1])])

        characteristics = {}

        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise
                continue

            mask = labels == cluster_id
            X_cluster = X[mask]

            logger.info(f"\nCluster {cluster_id}: {mask.sum()} events")

            # Compute mean and std for each feature
            means = X_cluster.mean(axis=0)
            stds = X_cluster.std(axis=0)

            # Find distinctive features (high absolute mean)
            top_features_idx = np.argsort(np.abs(means))[-10:][::-1]

            logger.info("  Top distinctive features:")
            top_feats = []
            for idx in top_features_idx:
                feat_name = feature_names[idx] if idx < len(feature_names) else f'feat_{idx}'
                logger.info(f"    {feat_name}: mean={means[idx]:.3f}, std={stds[idx]:.3f}")
                top_feats.append({
                    'name': feat_name,
                    'idx': int(idx),
                    'mean': float(means[idx]),
                    'std': float(stds[idx])
                })

            # Fault type distribution within cluster
            y_cluster = self.y_faults[mask]
            fault_dist = y_cluster.sum(axis=0)

            logger.info("  Fault type distribution:")
            for j, count in enumerate(fault_dist):
                if count > 0:
                    logger.info(f"    {self.fault_names[j]}: {int(count)} ({count/mask.sum()*100:.1f}%)")

            characteristics[int(cluster_id)] = {
                'size': int(mask.sum()),
                'top_features': top_feats,
                'fault_distribution': dict(zip(self.fault_names, fault_dist.astype(int).tolist()))
            }

        return characteristics

    def save_results(self):
        """Save all results."""
        output = {
            'all_faults': self.results.get('all_faults', {}),
            'per_fault': self.results.get('per_fault', {}),
            'characteristics': self.results.get('characteristics', {}),
            'known_subclasses': KNOWN_SUBCLASSES,
            'fault_names': self.fault_names,
            'n_total_events': len(self.X),
            'n_fault_events': len(self.X_faults)
        }

        output_file = self.output_dir / 'subclass_discovery.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(output, f, protocol=4)

        logger.info(f"\n✓ Saved: {output_file}")

        # Also save summary CSV
        summary_data = []
        if 'all_faults' in self.results:
            best = self.results['all_faults'].get('best_method', 'kmeans')
            labels = self.results['all_faults'].get(best, {}).get('labels', [])
            unique, counts = np.unique(labels, return_counts=True)
            for c, cnt in zip(unique, counts):
                summary_data.append({
                    'cluster_id': c,
                    'size': cnt,
                    'percentage': cnt / len(labels) * 100
                })

        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(self.output_dir / 'cluster_summary.csv', index=False)
            logger.info(f"✓ Saved: {self.output_dir / 'cluster_summary.csv'}")

    def run(self, target_k=19):
        """Run complete subclass discovery pipeline."""
        logger.info("="*70)
        logger.info("STEP 09b: ENHANCED SUBCLASS DISCOVERY")
        logger.info("="*70)

        self.load_data()

        # 1. Cluster all faults together
        self.results['all_faults'] = self.cluster_all_faults(target_k=target_k)

        # 2. Cluster per fault type
        self.results['per_fault'] = self.cluster_per_fault_type()

        # 3. Analyze characteristics of best clustering
        best_method = self.results['all_faults'].get('best_method', 'kmeans')
        best_labels = self.results['all_faults'][best_method]['labels']
        self.results['characteristics'] = self.analyze_cluster_characteristics(
            best_labels, method_name=best_method
        )

        # 4. Save results
        self.save_results()

        logger.info("\n" + "="*70)
        logger.info("STEP 09b COMPLETE!")
        logger.info("="*70)

        return self.results


def main():
    import os
    COOKED = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data'))
    INPUT_PATH = COOKED / 'features_engineered.pkl'
    OUTPUT_DIR = COOKED / 'step_09b_subclass_discovery'

    discovery = SubclassDiscovery(INPUT_PATH, OUTPUT_DIR)
    discovery.run(target_k=19)


if __name__ == '__main__':
    main()
