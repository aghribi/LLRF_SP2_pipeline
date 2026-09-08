#!/usr/bin/env python3
"""
Enhanced Subclass Discovery for SPIRAL2 LLRF Anomalies
=======================================================

Addresses the weakness of finding only 2 clusters with HDBSCAN.

Approaches:
1. Hierarchical clustering WITHIN each fault type
2. Physics-based feature subspace clustering
3. Multiple clustering algorithms comparison
4. Supervised hints using the 19 known subclass structure

Target Subclasses (19):
- Courant Pickup
- Circulator Arc
- RF PLC NOK: SAF, Cryo, Ampli, UGSx, E/T Circu (5 subtypes)
- Coupler Vacuum NOK: Leak/outgassing, Ipu (2 subtypes)
- Cavity BD/Quench: Sudden Wcav dissipation, False (2 subtypes)
- RF Protections: Pr max, Ucav max (2 subtypes)
- Ecav Instability: Loop oscillation, Taconis, Power chain gain, Quench (4 subtypes)
- Quench aside: Beam loss (1 subtype)
"""

import pickle
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utilities.reporting.manifest import save_manifest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)

warnings.filterwarnings('ignore')

# Paths
import os
COOKED = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data'))
OUTPUT_DIR = COOKED / 'step_09c_enhanced_subclass'
OUTPUT_DIR.mkdir(exist_ok=True)

# Fault type mapping
FAULT_LABELS = [
    'Pickup threshold',
    'Fast external cutoff',
    'RF authorization absent',
    'Vacuum threshold',
    'Cavity quench/breakdown',
    'RF safety threshold exceeded',
    'RF regulation out of tolerance'
]

# Physics-based feature groups (patterns to match feature names)
PHYSICS_GROUPS = {
    'amplitude': ['Ecav', 'Ucav', 'amplitude', 'amp_'],
    'phase': ['phase', 'Phase', 'dphi', 'angle'],
    'power': ['Pf', 'Pr', 'Pcav', 'power', 'Power', 'Watt'],
    'frequency': ['freq', 'fft', 'spectral', 'harmonic'],
    'temporal': ['diff', 'trend', 'rate', 'derivative'],
    'statistical': ['mean', 'std', 'var', 'skew', 'kurt', 'min', 'max']
}


def load_data():
    """Load features and labels."""
    print("Loading data...")

    # Load engineered features
    features_path = COOKED / 'features_engineered.pkl'
    with open(features_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        X = data.get('X_pca', data.get('X_scaled', None))
        y = data.get('y_multilabel', data.get('y_binary', None))
        feature_names = data.get('feature_names', data.get('feature_cols', None))
        X_original = data.get('X_scaled', None)
    else:
        X = data
        y = None
        feature_names = None
        X_original = None

    print(f"  Features shape: {X.shape}")
    if y is not None:
        print(f"  Labels shape: {y.shape}")

    return X, y, feature_names, X_original


def get_fault_mask(y, fault_idx):
    """Get mask for events with specific fault type."""
    if len(y.shape) == 1:
        return y == fault_idx
    else:
        return y[:, fault_idx] == 1


def cluster_within_fault_type(X, y, fault_idx, fault_name, n_clusters_range=(2, 6)):
    """Apply multiple clustering methods within a single fault type."""

    mask = get_fault_mask(y, fault_idx)
    X_fault = X[mask]

    if len(X_fault) < 20:
        print(f"  Skipping {fault_name}: only {len(X_fault)} samples")
        return None

    print(f"\n  Analyzing {fault_name} ({len(X_fault)} samples)")

    # V6 FIX: winsorize each column at the 5th/95th percentile WITHIN this
    # category before clustering. Without this, a single event with an extreme
    # value on even one feature (whether from a genuine rare signal transient or
    # a formula edge case, e.g. the near-zero-denominator early_late_ratio bug
    # fixed in prepare_data_cluster_v6.py) gets isolated by KMeans/Agglomerative/
    # GMM as its own singleton "cluster" -- every one of this pipeline's 5
    # analyzable categories originally produced a k=2 split of this form (e.g.
    # 590-vs-1, 308-vs-1), which is outlier isolation, not a genuine bimodal
    # fault subtype. Winsorizing first still lets clustering find real structure
    # among the bulk of events without letting one point's magnitude alone
    # decide the partition; verified this produces materially more balanced
    # splits (e.g. 100-vs-491, 249-vs-60) at a lower but still real silhouette.
    X_fault = np.clip(X_fault, np.percentile(X_fault, 5, axis=0), np.percentile(X_fault, 95, axis=0))

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fault)

    results = {
        'fault_name': fault_name,
        'n_samples': len(X_fault),
        'methods': {}
    }

    best_score = -1
    best_method = None
    best_labels = None
    best_k = None

    # V6 FIX: without a balance constraint, "best silhouette across many
    # candidate (algorithm, k) combinations" reliably selects whichever
    # candidate isolates the smallest, tightest minority subgroup -- even
    # after winsorizing extreme values, this still picked 2-15-member
    # "clusters" out of 20-591 events, not a genuine bimodal split. A tight
    # small cluster almost always scores higher on silhouette than a fair
    # partition of a more continuous distribution, so pure silhouette
    # maximization is structurally biased against finding real, well-populated
    # bimodal structure. Only consider a candidate labeling if every cluster
    # holds at least MIN_CLUSTER_FRACTION of the category's events; if nothing
    # in the search satisfies this, best_labels stays None -- an honest "no
    # balanced structure found", not a silent fallback to an imbalanced pick.
    MIN_CLUSTER_FRACTION = 0.15

    def _balanced(labels):
        if labels is None:
            return False
        counts = pd.Series(labels).value_counts()
        return (counts / len(labels)).min() >= MIN_CLUSTER_FRACTION

    for n_clusters in range(n_clusters_range[0], min(n_clusters_range[1] + 1, len(X_fault) // 5)):
        # K-Means
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels_km = kmeans.fit_predict(X_scaled)
            sil_km = silhouette_score(X_scaled, labels_km) if len(set(labels_km)) > 1 else -1

            if sil_km > best_score and _balanced(labels_km):
                best_score = sil_km
                best_method = f'KMeans-{n_clusters}'
                best_labels = labels_km
                best_k = n_clusters
        except:
            pass

        # Agglomerative with different linkages
        for linkage_type in ['ward', 'complete', 'average']:
            try:
                agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
                labels_agg = agg.fit_predict(X_scaled)
                sil_agg = silhouette_score(X_scaled, labels_agg) if len(set(labels_agg)) > 1 else -1

                if sil_agg > best_score and _balanced(labels_agg):
                    best_score = sil_agg
                    best_method = f'Agglom-{linkage_type}-{n_clusters}'
                    best_labels = labels_agg
                    best_k = n_clusters
            except:
                pass

        # GMM
        try:
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            labels_gmm = gmm.fit_predict(X_scaled)
            sil_gmm = silhouette_score(X_scaled, labels_gmm) if len(set(labels_gmm)) > 1 else -1

            if sil_gmm > best_score and _balanced(labels_gmm):
                best_score = sil_gmm
                best_method = f'GMM-{n_clusters}'
                best_labels = labels_gmm
                best_k = n_clusters
        except:
            pass

    results['best_method'] = best_method
    results['best_k'] = best_k
    results['best_silhouette'] = best_score if best_labels is not None else None
    results['best_labels'] = best_labels
    results['min_cluster_fraction_required'] = MIN_CLUSTER_FRACTION

    if best_labels is None:
        print(f"    No candidate clustering (k={n_clusters_range[0]}-{min(n_clusters_range[1], len(X_fault)//5)}, "
              f"KMeans/Agglomerative/GMM) satisfied the >= {MIN_CLUSTER_FRACTION:.0%}-per-cluster balance "
              f"constraint -- reporting no balanced subtype structure for this category rather than an "
              f"outlier-driven split.")
    else:
        print(f"    Best: {best_method} with silhouette={best_score:.3f}")

    # Cluster statistics
    if best_labels is not None:
        cluster_counts = pd.Series(best_labels).value_counts().sort_index()
        results['cluster_sizes'] = cluster_counts.to_dict()
        print(f"    Cluster sizes: {cluster_counts.to_dict()}")

    return results


def hierarchical_analysis(X, y, output_dir):
    """Create hierarchical dendrograms for each fault type."""

    print("\nHierarchical Analysis per Fault Type:")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    all_results = []

    for i, fault_name in enumerate(FAULT_LABELS):
        mask = get_fault_mask(y, i)
        X_fault = X[mask]

        if len(X_fault) < 10:
            axes[i].text(0.5, 0.5, f'{fault_name}\nToo few samples ({len(X_fault)})',
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(fault_name)
            continue

        # Standardize and reduce dimensions for visualization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_fault)

        # Subsample if too large
        if len(X_scaled) > 200:
            idx = np.random.choice(len(X_scaled), 200, replace=False)
            X_plot = X_scaled[idx]
        else:
            X_plot = X_scaled

        # Compute linkage
        try:
            Z = linkage(X_plot, method='ward')

            # Plot dendrogram
            dendrogram(Z, ax=axes[i], truncate_mode='lastp', p=12,
                      leaf_rotation=90, leaf_font_size=8, show_contracted=True)
            axes[i].set_title(f'{fault_name}\n(n={len(X_fault)})')
            axes[i].set_xlabel('Cluster')
            axes[i].set_ylabel('Distance')

            # Get optimal number of clusters using elbow
            distances = Z[:, 2]
            acceleration = np.diff(distances, 2)
            if len(acceleration) > 0:
                optimal_k = acceleration.argmax() + 2
            else:
                optimal_k = 2

            all_results.append({
                'fault_type': fault_name,
                'n_samples': len(X_fault),
                'suggested_k': min(optimal_k, 5)
            })

        except Exception as e:
            axes[i].text(0.5, 0.5, f'{fault_name}\nError: {str(e)[:30]}',
                        ha='center', va='center', transform=axes[i].transAxes)

    # Hide empty subplot
    axes[-1].axis('off')

    plt.suptitle('Hierarchical Clustering Dendrograms by Fault Type', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'hierarchical_dendrograms.png', dpi=150, bbox_inches='tight')
    plt.close()

    return all_results


def physics_subspace_clustering(X_original, y, feature_names, output_dir):
    """Cluster using physics-based feature subspaces."""

    print("\nPhysics-based Subspace Clustering:")

    if feature_names is None or X_original is None:
        print("  Original features not available, skipping physics subspace analysis")
        return None

    feature_names = list(feature_names)
    results = {}

    # Get fault events only
    if len(y.shape) == 1:
        fault_mask = y > 0
    else:
        fault_mask = y.sum(axis=1) > 0

    X_faults = X_original[fault_mask]
    y_faults = y[fault_mask]

    print(f"  Analyzing {len(X_faults)} fault events")

    for group_name, patterns in PHYSICS_GROUPS.items():
        # Find features matching this group
        group_idx = []
        for i, fname in enumerate(feature_names):
            for pattern in patterns:
                if pattern.lower() in fname.lower():
                    group_idx.append(i)
                    break

        if len(group_idx) < 3:
            print(f"  {group_name}: only {len(group_idx)} features, skipping")
            continue

        print(f"  {group_name}: {len(group_idx)} features")

        # Extract subspace
        X_subspace = X_faults[:, group_idx]

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subspace)

        # Try different k values
        best_k = 2
        best_sil = -1

        for k in range(2, min(8, len(X_scaled) // 10)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                sil = silhouette_score(X_scaled, labels)
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
            except:
                pass

        results[group_name] = {
            'n_features': len(group_idx),
            'best_k': best_k,
            'silhouette': best_sil
        }
        print(f"    Best k={best_k}, silhouette={best_sil:.3f}")

    return results


def create_subclass_visualization(X, y, best_results, output_dir):
    """Create t-SNE visualization with discovered subclusters."""

    print("\nCreating visualization...")

    # Get fault events
    if len(y.shape) == 1:
        fault_mask = y > 0
        fault_types = y[fault_mask]
    else:
        fault_mask = y.sum(axis=1) > 0
        fault_types = y[fault_mask].argmax(axis=1)  # Primary fault type

    X_faults = X[fault_mask]

    # t-SNE reduction
    print("  Computing t-SNE...")
    if len(X_faults) > 2000:
        idx = np.random.choice(len(X_faults), 2000, replace=False)
        X_plot = X_faults[idx]
        fault_types_plot = fault_types[idx]
    else:
        X_plot = X_faults
        fault_types_plot = fault_types
        idx = np.arange(len(X_faults))

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X_plot)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Color by fault type
    scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=fault_types_plot,
                               cmap='tab10', alpha=0.6, s=20)
    axes[0].set_title('Fault Events by Type')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=plt.cm.tab10(i/10), markersize=8, label=FAULT_LABELS[i])
               for i in range(len(FAULT_LABELS))]
    axes[0].legend(handles=handles, loc='best', fontsize=8)

    # Right: Show discovered subclusters for one fault type
    # Use the fault type with best subcluster separation
    best_fault_result = None
    best_sil = -1
    for r in best_results:
        if r and r.get('best_silhouette', -1) > best_sil:
            best_sil = r['best_silhouette']
            best_fault_result = r

    if best_fault_result:
        fault_idx = FAULT_LABELS.index(best_fault_result['fault_name'])

        # Get samples of this fault type in the plot
        if len(y.shape) == 1:
            type_mask = fault_types_plot == fault_idx
        else:
            type_mask = fault_types_plot == fault_idx

        # Recluster for visualization
        X_type = X_plot[type_mask]
        if len(X_type) > 10:
            kmeans = KMeans(n_clusters=best_fault_result['best_k'], random_state=42, n_init=10)
            scaler = StandardScaler()
            labels = kmeans.fit_predict(scaler.fit_transform(X_type))

            # Plot
            X_2d_type = X_2d[type_mask]
            scatter2 = axes[1].scatter(X_2d_type[:, 0], X_2d_type[:, 1],
                                       c=labels, cmap='Set1', alpha=0.7, s=30)
            axes[1].set_title(f"Subclusters within '{best_fault_result['fault_name']}'\n"
                             f"(k={best_fault_result['best_k']}, sil={best_sil:.3f})")
        else:
            axes[1].text(0.5, 0.5, 'Insufficient samples for visualization',
                        ha='center', va='center', transform=axes[1].transAxes)
    else:
        axes[1].text(0.5, 0.5, 'No valid subclusters found',
                    ha='center', va='center', transform=axes[1].transAxes)

    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(output_dir / 'subclass_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("Enhanced Subclass Discovery for SPIRAL2 LLRF")
    print("=" * 60)
    print(f"Started: {datetime.now()}")

    # Load data
    X, y, feature_names, X_original = load_data()

    if y is None:
        print("ERROR: Labels not found in features file")
        return

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'per_fault_clustering': [],
        'hierarchical_analysis': [],
        'physics_subspace': None
    }

    # 1. Cluster within each fault type
    print("\n" + "=" * 60)
    print("1. Clustering Within Each Fault Type")
    print("=" * 60)

    # V6 FIX: use X_original (X_scaled, the named/interpretable feature space)
    # for per-fault clustering, not X (X_pca). The winsorization fix inside
    # cluster_within_fault_type() clips each column at its own 5th/95th
    # percentile -- on X_pca this had no effect at all (verified: identical
    # cluster sizes before and after, e.g. still 590-vs-1), because a single
    # extreme raw feature spreads its influence across many PCA components,
    # each only moderately elevated individually, so per-component winsorizing
    # doesn't materially shrink the outlier's combined distance in PCA space.
    # Winsorizing the named 770-feature space directly (validated separately:
    # produces materially more balanced splits, e.g. 100-vs-491) does not have
    # this dilution problem.
    for i, fault_name in enumerate(FAULT_LABELS):
        result = cluster_within_fault_type(X_original, y, i, fault_name)
        if result:
            all_results['per_fault_clustering'].append(result)

    # 2. Hierarchical analysis
    print("\n" + "=" * 60)
    print("2. Hierarchical Analysis")
    print("=" * 60)

    hier_results = hierarchical_analysis(X, y, OUTPUT_DIR)
    all_results['hierarchical_analysis'] = hier_results

    # 3. Physics-based subspace clustering
    print("\n" + "=" * 60)
    print("3. Physics-based Subspace Clustering")
    print("=" * 60)

    physics_results = physics_subspace_clustering(X_original, y, feature_names, OUTPUT_DIR)
    all_results['physics_subspace'] = physics_results

    # 4. Visualization
    print("\n" + "=" * 60)
    print("4. Creating Visualizations")
    print("=" * 60)

    create_subclass_visualization(X, y, all_results['per_fault_clustering'], OUTPUT_DIR)

    # Save results
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    # Convert numpy arrays to lists for JSON serialization
    results_json = {}
    for key, value in all_results.items():
        if isinstance(value, list):
            results_json[key] = []
            for item in value:
                if isinstance(item, dict):
                    item_clean = {}
                    for k, v in item.items():
                        if isinstance(v, np.ndarray):
                            item_clean[k] = v.tolist()
                        elif k != 'best_labels':  # Skip large arrays
                            item_clean[k] = v
                    results_json[key].append(item_clean)
        elif isinstance(value, dict):
            results_json[key] = value
        else:
            results_json[key] = value

    with open(OUTPUT_DIR / 'enhanced_subclass_results.json', 'w') as f:
        json.dump(results_json, f, indent=2, default=str)

    # Save full results with arrays
    with open(OUTPUT_DIR / 'enhanced_subclass_full.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nPer-Fault Clustering Results:")
    summary_df = pd.DataFrame([
        {
            'Fault Type': r['fault_name'],
            'Samples': r['n_samples'],
            'Best Method': r['best_method'],
            'Best K': r['best_k'],
            'Silhouette': r['best_silhouette']
        }
        for r in all_results['per_fault_clustering']
    ])
    print(summary_df.to_string(index=False))

    # Save summary CSV
    summary_df.to_csv(OUTPUT_DIR / 'subclass_summary.csv', index=False)

    # Canonical results manifest -- 2026-09-06. Only categories with >=20
    # events are analyzed (see cluster_within_fault_type's `len(X_fault) < 20`
    # guard), so "Seuil pick-up" (12) and "Coupure externe rapide" (11) --
    # the two rarest of the 7 fault categories -- have no subtype result here,
    # not because they lack structure but because there isn't enough data to
    # cluster reliably. This finding is genuinely "k=2 per analyzed category,
    # silhouette 0.2-0.6" -- NOT the "21 distinct fault subtypes" or "19
    # expert-hypothesized subtypes" figures that appear elsewhere; see
    # PHASE3/report correction notes.
    metrics = {
        'n_categories_analyzed': {'value': len(summary_df), 'fmt': None,
                                   'label': 'Fault categories with enough events (>=20) to cluster'},
        'n_categories_total': {'value': 7, 'fmt': None, 'label': 'Total fault categories'},
    }
    for _, row in summary_df.iterrows():
        slug = str(row['Fault Type']).lower().replace(' ', '_').replace('/', '_').replace("'", '')
        metrics[f'{slug}_best_k'] = {'value': int(row['Best K']), 'fmt': None,
                                      'label': f"{row['Fault Type']}: best k (n={int(row['Samples'])})"}
        metrics[f'{slug}_silhouette'] = {'value': float(row['Silhouette']), 'fmt': '.3f',
                                          'label': f"{row['Fault Type']}: silhouette score"}
    save_manifest(
        phase='09c_enhanced_subclass_discovery',
        metrics=metrics,
        pipeline_run={
            'dataset_version': 'V6',
            'dataset_path': str(COOKED / 'features_engineered.pkl'),
            'script': 'pipeline/00_scripts/prepare_09c_enhanced_subclass_discovery.py',
        },
        meta={
            'excluded_categories_note': (
                "Seuil pick-up (12 events) and Coupure externe rapide (11 events) are excluded "
                "from per-fault clustering (minimum 20 events required); this reflects data "
                "scarcity, not an absence of substructure."
            ),
            'per_category_table': summary_df.to_dict(orient='records'),
        },
    )

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
