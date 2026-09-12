#!/usr/bin/env python3
"""
Step 07b: Phase 2 - Multi-Label Classification, Unsupervised

Description: Unsupervised counterpart to step 07's supervised multi-label
             classification (RF baseline) and Phases A/B/C (BR/CC/LP). Two
             families of methods, neither trained on the fault labels:

Part 1 -- Anomaly-detection scoring (step 05's 5 methods: Isolation Forest,
LOF, Mahalanobis distance, PCA reconstruction error, DBSCAN), each fit on
X_train (mixed classes, matching step 05/prepare_04a_dl_anomaly.py
convention) and scored on X_test. Per-fault-category one-vs-rest ROC AUC is
computed against each of the 7 fault_column_names columns of y_multilabel
independently (DBSCAN excluded from AUC, same as step 05's ensemble --
transductive, binary flag only).

Part 2 -- Clustering-vs-truth alignment (step 09's 3 clustering methods:
K-Means, Agglomerative/Hierarchical, GMM), fit on the FULL dataset (no
train/test split -- an unsupervised descriptive comparison, not a
generalization claim, matching step 09's own no-split precedent) at
k = number of true categories present. Cluster assignments are compared
against a single-label ground truth built the same way prepare_08_phase3a_
rootcause.py builds its y_root target: argmax(y_multilabel, axis=1) for
fault events (falls back to 'Normal' for non-fault events), NOT a
independently-verified root cause -- see that script's own caveat, reused
here unchanged. Alignment is reported via Adjusted Rand Index, Normalized
Mutual Information, and cluster purity.

2026-09-09: pretrigger toggle added (--pretrigger), restricting feature_cols
to leakage_safe_features.true_precursor_columns() before either part runs --
same fixed pre-trigger-only window used by step 05, step 06b, and the
supervised multi-label pretrigger variants. Completes the 2x2
supervised/unsupervised x full-window/pretrigger grid for the multi-label
task.

Input:  cooked_data/step_03_features/features_engineered.pkl
Output: cooked_data/step_07b_phase2_unsupervised/multilabel_unsupervised.pkl

Usage:
    python prepare_07b_phase2_multilabel_unsupervised.py \
        --input /path/to/step_03_features \
        --output /path/to/step_07b_phase2_unsupervised \
        [--pretrigger]
"""

import sys
import logging
from pathlib import Path
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA as SkPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, adjusted_rand_score,
                              normalized_mutual_info_score, silhouette_score)
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utilities.reporting.manifest import save_manifest

sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import leakage_safe_split, raw_feature_matrix, true_precursor_columns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler('step_07b_phase2_unsupervised.log'),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

ANOMALY_METHODS = ['isolation_forest', 'lof', 'mahalanobis', 'pca_reconstruction']
CLUSTER_METHODS = ['kmeans', 'agglomerative', 'gmm']


def _minmax(a):
    a = np.asarray(a, dtype=float)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _fit_score_anomaly_methods(X_train, X_test):
    """Fit the 4 continuous-score anomaly methods on X_train, score X_test.
    Returns {method_name: scores_array}. Mirrors step 05's exact hyperparameters."""
    scores = {}

    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    scores['isolation_forest'] = -iso_forest.decision_function(X_test)

    lof = LocalOutlierFactor(n_neighbors=20, contamination='auto', novelty=True, n_jobs=-1)
    lof.fit(X_train)
    scores['lof'] = -lof.decision_function(X_test)

    maha_pca = SkPCA(n_components=min(30, X_train.shape[0] - 1, X_train.shape[1]), random_state=42)
    X_train_maha = maha_pca.fit_transform(X_train)
    X_test_maha = maha_pca.transform(X_test)
    cov_estimator = LedoitWolf().fit(X_train_maha)
    scores['mahalanobis'] = cov_estimator.mahalanobis(X_test_maha)

    recon_pca = SkPCA(n_components=0.95, random_state=42)
    recon_pca.fit(X_train)
    X_test_recon = recon_pca.inverse_transform(recon_pca.transform(X_test))
    scores['pca_reconstruction'] = np.mean((X_test - X_test_recon) ** 2, axis=1)

    return scores


def _purity_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total = 0
    for c in np.unique(y_pred):
        mask = y_pred == c
        if mask.sum() == 0:
            continue
        _, counts = np.unique(y_true[mask], return_counts=True)
        total += counts.max()
    return total / len(y_true)


class Step07bProcessor:
    def __init__(self, input_dir, output_dir, pretrigger=False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pretrigger = pretrigger
        self.phase_suffix = '_pretrigger' if pretrigger else ''
        self.dataset_version = (lambda _n: _n.upper() if _n else 'V2')(
            self.input_dir.name.replace('cooked_data', '').lstrip('_'))

    def load_inputs(self):
        with open(self.input_dir / 'features_engineered.pkl', 'rb') as f:
            self.data = pickle.load(f)
        if self.pretrigger:
            self.data['feature_cols'] = true_precursor_columns(self.data['feature_cols'])
        logger.info(f"Loaded: {self.data['features_all'].shape}, "
                    f"{len(self.data['feature_cols'])} filtered feature columns "
                    f"(pretrigger={self.pretrigger})")

    def _anomaly_detection_part(self, y_multilabel, fault_names):
        logger.info("="*70)
        logger.info("PART 1: ANOMALY-DETECTION SCORING (one-vs-rest per fault category)")
        logger.info("="*70)

        split = leakage_safe_split(self.data, y_multilabel, test_size=0.3, random_state=42,
                                    stratify=False)
        X_train, X_test = split['X_train'], split['X_test']
        y_test = split['y_test']

        scores = _fit_score_anomaly_methods(X_train, X_test)

        # Headline (single-split, seed=42) per-category AUC for the 4 continuous methods.
        headline_auc = {m: {} for m in ANOMALY_METHODS}
        for m in ANOMALY_METHODS:
            for i, fname in enumerate(fault_names):
                y_col = y_test[:, i]
                if len(np.unique(y_col)) == 2:
                    headline_auc[m][fname] = float(roc_auc_score(y_col, scores[m]))
                else:
                    headline_auc[m][fname] = None
            logger.info(f"  {m}: " + ", ".join(
                f"{k}={v:.3f}" if v is not None else f"{k}=n/a" for k, v in headline_auc[m].items()))

        # DBSCAN (transductive, fit+predict on X_test directly): binary outlier
        # flag only, no AUC -- same exclusion as step 05's ensemble.
        # 2026-09-09: eps auto-tuned via the same k-distance percentile
        # heuristic prepare_09_phase3b_clustering.py already uses for its own
        # DBSCAN (5-NN distance, 75th percentile), instead of the fixed
        # eps=0.5 carried over from step 05 -- that fixed value was tuned for
        # step 05's own feature space and flagged 100% of test events as
        # outliers here (confirmed degenerate on the first V7 run of this
        # script), i.e. it was uninformative, not a real finding.
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors.fit(X_test)
        nn_distances, _ = neighbors.kneighbors(X_test)
        dbscan_eps = float(np.percentile(nn_distances[:, -1], 75))
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=5, n_jobs=-1)
        dbscan_labels = dbscan.fit_predict(X_test)
        dbscan_outlier = (dbscan_labels == -1).astype(int)
        dbscan_recall = {}
        for i, fname in enumerate(fault_names):
            y_col = y_test[:, i]
            n_pos = int(y_col.sum())
            dbscan_recall[fname] = float(dbscan_outlier[y_col == 1].mean()) if n_pos > 0 else None
        logger.info(f"  dbscan (eps={dbscan_eps:.3f}, auto-tuned 75th-pct 5-NN distance) outlier_recall: " + ", ".join(
            f"{k}={v:.3f}" if v is not None else f"{k}=n/a" for k, v in dbscan_recall.items()))
        logger.info(f"  dbscan overall outlier fraction: {dbscan_outlier.mean():.3f}")

        # Repeated-split stability (10 repeats, different random_state each) --
        # same rigor as every other split-sensitive number in this pipeline
        # (step 05/07/phaseA-C/rootcause). AUC (not .predict()) needs a custom
        # loop -- repeated_leakage_safe_eval() assumes a .predict()-based metric.
        logger.info("\nRepeated-split stability check (10 splits, different random_state each)...")
        n_repeats = 10
        repeat_auc = {m: {fname: [] for fname in fault_names} for m in ANOMALY_METHODS}
        for r in range(n_repeats):
            split_r = leakage_safe_split(self.data, y_multilabel, test_size=0.3,
                                          random_state=100 + r, stratify=False)
            scores_r = _fit_score_anomaly_methods(split_r['X_train'], split_r['X_test'])
            y_test_r = split_r['y_test']
            for m in ANOMALY_METHODS:
                for i, fname in enumerate(fault_names):
                    y_col = y_test_r[:, i]
                    if len(np.unique(y_col)) == 2:
                        repeat_auc[m][fname].append(float(roc_auc_score(y_col, scores_r[m])))

        stability_auc = {m: {} for m in ANOMALY_METHODS}
        for m in ANOMALY_METHODS:
            for fname in fault_names:
                vals = repeat_auc[m][fname]
                if vals:
                    stability_auc[m][fname] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                                                'n_splits_with_both_classes': len(vals)}
                else:
                    stability_auc[m][fname] = None
            logger.info(f"  {m} (stability): " + ", ".join(
                f"{k}={v['mean']:.3f}+/-{v['std']:.3f}" if v is not None else f"{k}=n/a"
                for k, v in stability_auc[m].items()))

        return {
            'headline_auc': headline_auc,
            'dbscan_eps': dbscan_eps,
            'dbscan_outlier_recall': dbscan_recall,
            'dbscan_overall_outlier_fraction': float(dbscan_outlier.mean()),
            'stability_auc': stability_auc,
            'n_test': int(len(y_test)),
        }

    def _clustering_alignment_part(self, y_multilabel, fault_names, y_binary):
        logger.info("="*70)
        logger.info("PART 2: CLUSTERING-VS-TRUTH ALIGNMENT")
        logger.info("="*70)

        # Same convention as prepare_08_phase3a_rootcause.py's y_root: argmax
        # of the multi-hot encoding = "first flagged category in bit order",
        # a deterministic simplification, not an independently-verified
        # ground truth -- see that script's own caveat, reused unchanged here.
        # Extended to non-fault events as category 'Normal'.
        has_fault = y_multilabel.sum(axis=1) > 0
        fault_idx = np.argmax(y_multilabel, axis=1)
        category_names = ['Normal'] + list(fault_names)
        category_idx = np.where(has_fault, fault_idx + 1, 0)

        uniq, counts = np.unique(category_idx, return_counts=True)
        logger.info("Category distribution (argmax-of-multilabel convention): " + ", ".join(
            f"{category_names[u]}={c}" for u, c in zip(uniq, counts)))

        X_raw, _ = raw_feature_matrix(self.data)
        scaler = StandardScaler()
        X_all = scaler.fit_transform(X_raw)
        k = len(uniq)
        logger.info(f"Clustering all {len(X_all)} events into k={k} clusters "
                    f"(= number of distinct categories present)")

        results = {}
        for method in CLUSTER_METHODS:
            if method == 'kmeans':
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(X_all)
            elif method == 'agglomerative':
                model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(X_all)
            else:
                model = GaussianMixture(n_components=k, random_state=42)
                labels = model.fit_predict(X_all)

            ari = float(adjusted_rand_score(category_idx, labels))
            nmi = float(normalized_mutual_info_score(category_idx, labels))
            purity = float(_purity_score(category_idx, labels))
            silh = float(silhouette_score(X_all, labels)) if len(np.unique(labels)) > 1 else None
            logger.info(f"  {method}: ARI={ari:.4f} NMI={nmi:.4f} purity={purity:.4f} "
                        f"silhouette={silh:.4f}" if silh is not None else
                        f"  {method}: ARI={ari:.4f} NMI={nmi:.4f} purity={purity:.4f} silhouette=n/a")
            results[method] = {'ari': ari, 'nmi': nmi, 'purity': purity, 'silhouette': silh, 'k': k}

        return {'per_method': results, 'k': k, 'category_names': category_names,
                'category_counts': {category_names[u]: int(c) for u, c in zip(uniq, counts)}}

    def process(self):
        y_multilabel = self.data['y_multilabel']
        y_binary = self.data.get('y_binary')
        fault_names = self.data['fault_column_names']
        if y_multilabel is None:
            raise ValueError("No multi-label data available")

        anomaly_results = self._anomaly_detection_part(y_multilabel, fault_names)
        cluster_results = self._clustering_alignment_part(y_multilabel, fault_names, y_binary)

        # Flatten into a manifest-friendly metrics dict.
        metrics = {}
        for m in ANOMALY_METHODS:
            for fname in fault_names:
                v = anomaly_results['headline_auc'][m][fname]
                key = f'{m}_auc_{fname}'
                if v is not None:
                    metrics[key] = {'value': v, 'fmt': '.4f',
                                     'label': f'{m} one-vs-rest AUC vs. {fname} (single split, seed=42)'}
            for fname in fault_names:
                sv = anomaly_results['stability_auc'][m][fname]
                key = f'{m}_auc_stability_{fname}'
                if sv is not None:
                    metrics[f'{key}_mean'] = {'value': sv['mean'], 'fmt': '.4f',
                                               'label': f'{m} one-vs-rest AUC vs. {fname} (10-split mean)'}
                    metrics[f'{key}_std'] = {'value': sv['std'], 'fmt': '.4f',
                                              'label': f'{m} one-vs-rest AUC vs. {fname} (10-split std)'}
        metrics['dbscan_eps'] = {
            'value': anomaly_results['dbscan_eps'], 'fmt': '.4f',
            'label': 'DBSCAN eps (auto-tuned: 75th-pct of 5-NN distances on X_test)'}
        metrics['dbscan_overall_outlier_fraction'] = {
            'value': anomaly_results['dbscan_overall_outlier_fraction'], 'fmt': '.3f',
            'label': 'DBSCAN: fraction of test events flagged as outliers'}
        for method in CLUSTER_METHODS:
            r = cluster_results['per_method'][method]
            metrics[f'{method}_ari'] = {'value': r['ari'], 'fmt': '.4f', 'label': f'{method} ARI vs. primary category'}
            metrics[f'{method}_nmi'] = {'value': r['nmi'], 'fmt': '.4f', 'label': f'{method} NMI vs. primary category'}
            metrics[f'{method}_purity'] = {'value': r['purity'], 'fmt': '.4f', 'label': f'{method} purity vs. primary category'}
            if r['silhouette'] is not None:
                metrics[f'{method}_silhouette'] = {'value': r['silhouette'], 'fmt': '.4f', 'label': f'{method} internal silhouette'}
        metrics['clustering_k'] = {'value': cluster_results['k'], 'fmt': None, 'label': 'Number of clusters (= number of distinct categories present)'}
        metrics['n_test_anomaly_part'] = {'value': anomaly_results['n_test'], 'fmt': None, 'label': 'Test-set events (Part 1, anomaly detection)'}

        save_manifest(
            phase=f'07b_phase2_multilabel_unsupervised{self.phase_suffix}',
            metrics=metrics,
            pipeline_run={
                'dataset_version': self.dataset_version,
                'dataset_path': str(self.input_dir / 'features_engineered.pkl'),
                'script': 'pipeline/00_scripts/prepare_07b_phase2_multilabel_unsupervised.py',
            },
            meta={
                'pretrigger': self.pretrigger,
                'anomaly_detection_methods': ANOMALY_METHODS + ['dbscan (outlier flag only, no AUC)'],
                'clustering_methods': CLUSTER_METHODS,
                'category_counts': cluster_results['category_counts'],
                'ground_truth_caveat': (
                    "Cluster-alignment ground truth (Part 2) is argmax(y_multilabel, axis=1), "
                    "extended with a 'Normal' category for non-fault events -- the same "
                    "deterministic 'first flagged category in bit order' convention as "
                    "prepare_08_phase3a_rootcause.py's y_root, not an independently-verified "
                    "root cause. Multi-fault events are folded into their first-flagged "
                    "category only."
                ),
                'auc_scope_note': (
                    "Part 1 AUCs are one-vs-rest per fault category (each event's OTHER labels "
                    "are ignored for that column's AUC), matching the multi-label task's own "
                    "per-label evaluation convention used in step 07/Phase A-C, not a "
                    "single-label multiclass comparison."
                ),
            },
        )

        return {'anomaly_detection': anomaly_results, 'clustering_alignment': cluster_results}

    def save_outputs(self, results):
        output_file = self.output_dir / 'multilabel_unsupervised.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f, protocol=4)
        logger.info(f"\n✓ Saved: {output_file}")

    def run(self):
        logger.info("STEP 07b: PHASE 2 - MULTI-LABEL CLASSIFICATION, UNSUPERVISED")
        self.load_inputs()
        results = self.process()
        self.save_outputs(results)
        logger.info("STEP 07b COMPLETE!")
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pretrigger', action='store_true',
                         help='Restrict to the pre-trigger-only true-precursor feature subset')
    args = parser.parse_args()
    Step07bProcessor(args.input, args.output, pretrigger=args.pretrigger).run()


if __name__ == '__main__':
    main()
