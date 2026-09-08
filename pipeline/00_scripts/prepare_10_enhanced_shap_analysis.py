#!/usr/bin/env python3
"""
Enhanced SHAP Analysis for SPIRAL2 LLRF Anomalies
==================================================

Addresses the weakness of limited explainability by providing:
1. Per-label SHAP analysis (one analysis per fault type)
2. Physics-grouped feature importance
3. Feature interaction analysis
4. Comparative feature importance across fault types
"""

import pickle
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import leakage_safe_split

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP not available, using feature importance instead")

# Paths
import os
# V6 FIX: this defaulted to the pre-V6 'cooked_data' (V2, 4509 events) if
# SPIRAL2_COOKED_DIR wasn't set -- a real reproducibility trap, not
# hypothetical: it silently fired during this session's own review-response
# work (importing this module before setting the env var reused Python's
# cached module-level COOKED path from the first import). Defaulting to the
# current, correct V6 directory instead; override via SPIRAL2_COOKED_DIR only
# to intentionally target a different dataset version.
COOKED = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6'))
OUTPUT_DIR = COOKED / 'step_10_enhanced_shap'
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

# V6 FIX: the original PHYSICS_GROUPS below classified by the FIRST matching
# substring in insertion order (e.g. 'Ucav' before 'std'), which put whole-
# signal Statistical features like Ucav_mean/Ucav_std into "Amplitude" instead
# of "Statistical", put Detuning/Microphonics/RF-Mismatch/Modulator-Command
# features (which Appendix A documents as their own physics families) into
# whichever of 6 generic buckets happened to match first, and would have
# classified a metadata field like meta_AMPT as "Amplitude" -- a physically
# suggestive label for what would actually be a non-physics leak-risk field.
# Replaced with an exact/prefix classifier built directly from
# engineer_features()'s real code structure (verified: 100% coverage, zero
# "Other" fallback, against the real 770-column V6 feature set), matching the
# seven physics families in Appendix A plus Statistical/Temporal/Metadata.
import re as _re

PHYSICS_FAMILY_EXACT = {
    'effective_decay_tau': 'Effective Decay', 'decay_rate': 'Effective Decay',
    'decay_fit_r2': 'Effective Decay', 'decay_non_exponentiality': 'Effective Decay',
    'decay_relative_to_nominal': 'Effective Decay', 'valid_decay_window': 'Effective Decay',
    'rf_mismatch_rms_pre': 'RF Mismatch', 'rf_mismatch_rms_post': 'RF Mismatch',
    'rf_mismatch_max_excursion': 'RF Mismatch', 'rf_mismatch_settling_time': 'RF Mismatch',
    'control_settling_time': 'RF Mismatch',  # alias of rf_mismatch_settling_time (see prepare_data_cluster_v6.py:969)
    'reflected_saturation_fraction': 'RF Mismatch',
    'control_saturation_fraction': 'RF Mismatch',  # alias of reflected_saturation_fraction (prepare_data_cluster_v6.py) -- newly survives correlation-dedup post-V6-fix since two constant-valued columns correlate as NaN, not 1.0
    'phase_std_pre': 'RF Mismatch', 'phase_std_post': 'RF Mismatch',  # co-located cavity-phase jitter, computed in the RF-mismatch block
    'phase_jitter_rms': 'Phase Stability', 'phase_noise_slope': 'Phase Stability',
    'phase_excursion_pp': 'Phase Stability', 'phase_psd_integral': 'Phase Stability',
    'forward_power_mean': 'RF Power Flow', 'reflected_power_mean': 'RF Power Flow',
    'power_imbalance_rms': 'RF Power Flow', 'power_jump_at_trigger': 'RF Power Flow',
    'power_reflection_ratio': 'RF Power Flow',
    # 'beam_present' (V6 and earlier) / 'feed_forward_enabled' (V7+, 2026-09-09 rename --
    # same underlying header field, was mislabeled as beam presence, see
    # prepare_data_cluster_v7.py's module docstring): kept both keys so this taxonomy
    # works unchanged against either dataset version.
    'beam_present': 'Metadata/Status', 'feed_forward_enabled': 'Metadata/Status',
    'llrf_loop_closed': 'Metadata/Status', 'rf_drive_on': 'Metadata/Status',
}
PHYSICS_FAMILY_PREFIX = {
    'detuning_': 'Detuning', 'microphonics_': 'Microphonics',
    'mod_amplitude_': 'Modulator Command', 'mod_phase_': 'Modulator Command',
    'meta_': 'Metadata/Status',
}
STATISTICAL_SUFFIXES = ('_mean', '_std', '_skewness', '_kurtosis', '_min', '_max', '_range',
                         '_median', '_q1', '_q3', '_iqr', '_rms', '_crest_factor',
                         '_peak_count', '_energy', '_zero_crossings', '_autocorr_lag1')
TEMPORAL_SUFFIXES = ('_trend_slope', '_trend_acceleration', '_cusum_max', '_cusum_min',
                      '_early_late_diff', '_early_late_ratio', '_var_ratio_late_early',
                      '_velocity_mean', '_velocity_std', '_velocity_max',
                      '_accel_mean', '_accel_std', '_accel_max')
_SEGMENT_PATTERN = _re.compile(r'_segment_(early|mid_early|mid|mid_late|late)_(mean|std|slope)$')


def load_data():
    """Load features and labels."""
    print("Loading data...")

    features_path = COOKED / 'features_engineered.pkl'
    with open(features_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        y = data.get('y_multilabel', data.get('y_binary', None))
        feature_names = list(data['feature_cols'])
    else:
        raise ValueError("Expected a features_engineered*.pkl dict with 'features_all'/'feature_cols'")

    print(f"  Features shape: {data['features_all'].shape}")
    print(f"  Labels shape: {y.shape if y is not None else 'N/A'}")
    print(f"  Feature names: {len(feature_names)}")

    # Return the raw pkl dict (not a pre-fit X_scaled) -- leakage_safe_split
    # fits scaler/PCA per-label, on each label's own train fold only.
    return data, y, feature_names


def get_physics_group(feature_name):
    """Classify a feature by the section of engineer_features() that produced
    it (exact name / prefix / suffix, verified 100% coverage against the real
    V6 feature set), not by a substring heuristic."""
    if feature_name in PHYSICS_FAMILY_EXACT:
        return PHYSICS_FAMILY_EXACT[feature_name]
    for prefix, group in PHYSICS_FAMILY_PREFIX.items():
        if feature_name.startswith(prefix):
            return group
    if _SEGMENT_PATTERN.search(feature_name):
        return 'Temporal'
    for suf in TEMPORAL_SUFFIXES:
        if feature_name.endswith(suf):
            return 'Temporal'
    for suf in STATISTICAL_SUFFIXES:
        if feature_name.endswith(suf):
            return 'Statistical'
    return 'Other'


def train_per_label_models(pkl_data, y, feature_names):
    """Train a model for each fault label and compute feature importance."""

    print("\nTraining per-label models...")

    # Get fault events only
    if len(y.shape) == 1:
        fault_mask = y > 0
        # Convert to multi-label format
        y_multi = np.zeros((len(y), len(FAULT_LABELS)))
        for i in range(len(FAULT_LABELS)):
            y_multi[:, i] = (y == i + 1).astype(int)
    else:
        fault_mask = y.sum(axis=1) > 0
        y_multi = y

    y_faults = y_multi[fault_mask]

    # Subset to fault events FIRST (raw, unscaled) -- scaler/PCA are fit per
    # label below, on each label's own train fold only.
    fault_data = {
        'features_all': pkl_data['features_all'].loc[fault_mask].reset_index(drop=True),
        'feature_cols': pkl_data['feature_cols'],
    }

    results = {}

    for i, fault_name in enumerate(FAULT_LABELS):
        y_label = y_faults[:, i] if len(y_faults.shape) > 1 else (y_faults == i + 1).astype(int)

        n_positive = y_label.sum()
        if n_positive < 10:
            print(f"  {fault_name}: only {n_positive} positive samples, skipping")
            continue

        print(f"  {fault_name}: {n_positive} positive samples")

        # Leakage-safe split: scaler fit on THIS label's train fold only.
        split = leakage_safe_split(fault_data, y_label, test_size=0.2, random_state=42,
                                    stratify=True)
        X_train, X_test = split['X_train'], split['X_test']
        y_train, y_test = split['y_train'], split['y_test']

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                    class_weight='balanced')
        rf.fit(X_train, y_train)

        # Feature importance from RF
        importance = rf.feature_importances_

        # Store results
        results[fault_name] = {
            'model': rf,
            'importance': importance,
            'n_positive': int(n_positive),
            'test_score': rf.score(X_test, y_test),
            'X_train': X_train,
            'X_test': X_test
        }

        print(f"    Test accuracy: {results[fault_name]['test_score']:.3f}")

    return results


def compute_shap_values(results, feature_names, max_samples=500):
    """Compute SHAP values for each label's model."""

    if not SHAP_AVAILABLE:
        print("\nSHAP not available, using RF feature importance instead")
        return None

    print("\nComputing SHAP values per label...")

    shap_results = {}

    for fault_name, data in results.items():
        print(f"  {fault_name}...")

        model = data['model']
        X_test = data['X_test']

        # Subsample if needed
        if len(X_test) > max_samples:
            idx = np.random.choice(len(X_test), max_samples, replace=False)
            X_explain = X_test[idx]
        else:
            X_explain = X_test

        try:
            # Use TreeExplainer for Random Forest
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)

            # For binary classification, shap_values might be a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Take positive class

            shap_results[fault_name] = {
                'shap_values': shap_values,
                'X_explain': X_explain,
                'expected_value': explainer.expected_value
            }

            print(f"    SHAP values shape: {shap_values.shape}")

        except Exception as e:
            print(f"    Error computing SHAP: {str(e)}")

    return shap_results


def create_per_label_importance_plot(results, feature_names, output_dir):
    """Create feature importance plots for each label."""

    print("\nCreating per-label importance plots...")

    n_labels = len(results)
    n_cols = 2
    n_rows = (n_labels + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for idx, (fault_name, data) in enumerate(results.items()):
        importance = data['importance']

        # Get top 15 features
        top_idx = np.argsort(importance)[-15:][::-1]
        top_names = [feature_names[i] if i < len(feature_names) else f'F{i}' for i in top_idx]
        top_importance = importance[top_idx]

        # Color by physics group
        all_groups = GROUP_NAMES
        colors = [plt.cm.Set2(all_groups.index(get_physics_group(name)) % 8)
                  for name in top_names]

        axes[idx].barh(range(15), top_importance[::-1], color=colors[::-1])
        axes[idx].set_yticks(range(15))
        axes[idx].set_yticklabels(top_names[::-1], fontsize=8)
        axes[idx].set_xlabel('Importance')
        axes[idx].set_title(f'{fault_name}\n(n={data["n_positive"]}, acc={data["test_score"]:.2f})')

    # Hide unused axes
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')

    # Add legend for physics groups
    all_groups = GROUP_NAMES
    handles = [plt.Rectangle((0, 0), 1, 1, color=plt.cm.Set2(i % 8))
               for i in range(len(all_groups))]
    fig.legend(handles, all_groups, loc='lower right',
               ncol=4, fontsize=10, title='Physics Groups')

    plt.suptitle('Feature Importance by Fault Type', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_dir / 'per_label_importance.png', dpi=150, bbox_inches='tight')
    plt.close()


GROUP_NAMES = ['Effective Decay', 'Detuning', 'Microphonics', 'RF Mismatch', 'Modulator Command',
               'Phase Stability', 'RF Power Flow', 'Statistical', 'Temporal', 'Metadata/Status', 'Other']


def create_physics_group_summary(shap_results, feature_names, output_dir):
    """Summarize physics-group importance for each fault type, from REAL SHAP values.

    V6 FIX (round 2 of paper review): this function previously received
    `results` (per-label RandomForestClassifier.feature_importances_, i.e.
    Gini impurity importance) and reported it under every "SHAP" label in the
    report -- the manifest phase name, every \\Metric...PhysicsGroupShap...
    macro, the table caption, and the abstract's headline claim -- despite
    `compute_shap_values()` computing genuine SHAP values elsewhere in this
    script that were never wired into this function. Found by an independent
    ML-methodology review pass; fixed by taking `shap_results` (real
    shap.TreeExplainer output) instead, and using mean(|SHAP value|) per
    feature -- the standard SHAP importance statistic -- as `importance`.

    Also reports both the group SUM (% of total importance) and the
    per-feature MEAN within each group (the physics-informed families total
    only \\Metric...TotalPhysicsFeatures{} features combined, versus 256
    Statistical and 466 Temporal -- a group-sum comparison alone structurally
    favors whichever group has more member features). A second review pass
    found the per-feature MEAN itself can be dominated by a single outlier
    feature in the small (4-8 feature) physics families (up to ~95% of a
    group's own sum from one feature in some categories) -- so this also
    reports the per-feature MEDIAN, which is far more robust to that failure
    mode, alongside the mean, rather than presenting the mean alone as if it
    were bias-free.
    """

    print("\nCreating physics group summary (from real SHAP values)...")

    feature_groups = [get_physics_group(name) for name in feature_names]
    group_counts = {g: feature_groups.count(g) for g in GROUP_NAMES}

    group_importance_pct = {}
    group_importance_mean = {}
    group_importance_median = {}

    for fault_name, data in shap_results.items():
        shap_values = np.asarray(data['shap_values'])
        if shap_values.ndim == 3:
            # (n_samples, n_features, n_classes) -- take the positive class.
            shap_values = shap_values[:, :, 1]
        importance = np.abs(shap_values).mean(axis=0)  # mean |SHAP| per feature, standard SHAP importance

        group_sums = {}
        group_means = {}
        group_medians = {}
        for group in GROUP_NAMES:
            group_mask = [g == group for g in feature_groups]
            if any(group_mask):
                vals = importance[group_mask]
                group_sums[group] = vals.sum()
                group_means[group] = vals.mean()
                group_medians[group] = np.median(vals)
            else:
                group_sums[group] = 0.0
                group_means[group] = 0.0
                group_medians[group] = 0.0

        total = sum(group_sums.values())
        if total > 0:
            group_importance_pct[fault_name] = {k: v / total * 100 for k, v in group_sums.items()}
        else:
            group_importance_pct[fault_name] = group_sums
        # Per-feature-mean/median values normalized to percent-of-total, so
        # all tables are on comparable (percentage) scales despite the very
        # different feature counts per group.
        mean_total = sum(group_means.values())
        if mean_total > 0:
            group_importance_mean[fault_name] = {k: v / mean_total * 100 for k, v in group_means.items()}
        else:
            group_importance_mean[fault_name] = group_means
        median_total = sum(group_medians.values())
        if median_total > 0:
            group_importance_median[fault_name] = {k: v / median_total * 100 for k, v in group_medians.items()}
        else:
            group_importance_median[fault_name] = group_medians

    df = pd.DataFrame(group_importance_pct).T[GROUP_NAMES]
    df_mean = pd.DataFrame(group_importance_mean).T[GROUP_NAMES]
    df_median = pd.DataFrame(group_importance_median).T[GROUP_NAMES]

    # Plot heatmap (sum-based, unchanged visualization; per-feature-mean/median saved to CSV only)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'SHAP importance (%, group sum)'})
    ax.set_title('SHAP Feature Importance by Physics Group and Fault Type (group sum)', fontsize=14)
    ax.set_xlabel('Physics Feature Group')
    ax.set_ylabel('Fault Type')
    plt.tight_layout()
    plt.savefig(output_dir / 'physics_group_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    df.to_csv(output_dir / 'physics_group_importance.csv')
    df_mean.to_csv(output_dir / 'physics_group_importance_per_feature_mean.csv')
    df_median.to_csv(output_dir / 'physics_group_importance_per_feature_median.csv')
    pd.Series(group_counts).to_csv(output_dir / 'physics_group_feature_counts.csv', header=['n_features'])

    return df, df_mean, df_median, group_counts


def create_comparative_analysis(shap_results, feature_names, output_dir):
    """Compare which features are important across multiple fault types.

    V6 FIX (found during the pipeline-bug fix/rerun, 2026-09-07): this
    function previously took `results` and used
    RandomForestClassifier.feature_importances_ (Gini/MDI) as `importance`,
    saved to all_feature_importance.csv and cited in the report's "top
    individual features" example as if it were SHAP -- the same
    Gini-mislabeled-as-SHAP bug already fixed once in
    create_physics_group_summary() (see its own docstring), just missed in
    this second, separate function. Fixed the same way: take `shap_results`
    (real shap.TreeExplainer output) and use mean(|SHAP value|) per feature.
    """

    print("\nCreating comparative analysis (from real SHAP values)...")

    # Get importance for all features across all labels
    all_importance = np.zeros((len(feature_names), len(shap_results)))
    fault_names_ordered = list(shap_results.keys())

    for i, (fault_name, data) in enumerate(shap_results.items()):
        shap_values = np.asarray(data['shap_values'])
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        all_importance[:, i] = np.abs(shap_values).mean(axis=0)

    # Find features important for multiple fault types
    # Threshold: top 10% of importance for each label
    thresholds = np.percentile(all_importance, 90, axis=0)
    important_mask = all_importance > thresholds

    # Count how many fault types each feature is important for
    importance_count = important_mask.sum(axis=1)

    # Features important for 3+ fault types
    multi_important_idx = np.where(importance_count >= 3)[0]

    if len(multi_important_idx) > 0:
        print(f"\n  Features important for 3+ fault types:")
        for idx in multi_important_idx[:10]:  # Top 10
            fname = feature_names[idx] if idx < len(feature_names) else f'Feature_{idx}'
            print(f"    {fname}: {importance_count[idx]} fault types")

    # Create figure: feature importance across fault types
    fig, ax = plt.subplots(figsize=(14, 10))

    # Select top 20 most consistently important features
    mean_importance = all_importance.mean(axis=1)
    top_idx = np.argsort(mean_importance)[-20:][::-1]

    data_plot = all_importance[top_idx, :]
    feature_names_plot = [feature_names[i] if i < len(feature_names) else f'F{i}' for i in top_idx]

    sns.heatmap(data_plot, annot=False, cmap='viridis', ax=ax,
                xticklabels=fault_names_ordered, yticklabels=feature_names_plot,
                cbar_kws={'label': 'Importance'})
    ax.set_title('Top 20 Features: Importance Across Fault Types', fontsize=14)
    ax.set_xlabel('Fault Type')
    ax.set_ylabel('Feature')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save detailed results
    importance_df = pd.DataFrame(
        all_importance,
        index=feature_names,
        columns=fault_names_ordered
    )
    importance_df['mean'] = importance_df.mean(axis=1)
    importance_df['std'] = importance_df.std(axis=1)
    importance_df['n_important'] = importance_count
    importance_df = importance_df.sort_values('mean', ascending=False)
    importance_df.to_csv(output_dir / 'all_feature_importance.csv')

    return importance_df


def create_shap_summary_plots(shap_results, feature_names, output_dir):
    """Create SHAP summary plots for each label."""

    if shap_results is None:
        print("\nSkipping SHAP plots (SHAP not available)")
        return

    print("\nCreating SHAP summary plots...")

    n_labels = len(shap_results)

    for fault_name, data in shap_results.items():
        shap_values = data['shap_values']
        X_explain = data['X_explain']

        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_explain, feature_names=feature_names,
                         show=False, max_display=15)
        plt.title(f'SHAP Summary: {fault_name}')
        plt.tight_layout()

        # Clean filename
        fname = fault_name.replace('/', '_').replace(' ', '_')
        plt.savefig(output_dir / f'shap_summary_{fname}.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    print("=" * 60)
    print("Enhanced SHAP Analysis for SPIRAL2 LLRF")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"SHAP available: {SHAP_AVAILABLE}")

    # Load data
    pkl_data, y, feature_names = load_data()

    if y is None:
        print("ERROR: Labels not found")
        return

    # Train per-label models
    results = train_per_label_models(pkl_data, y, feature_names)

    if not results:
        print("ERROR: No models trained")
        return

    # Compute SHAP values
    shap_results = compute_shap_values(results, feature_names)

    # Create visualizations
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)

    create_per_label_importance_plot(results, feature_names, OUTPUT_DIR)

    if not shap_results:
        raise SystemExit(
            "ERROR: SHAP computation failed or SHAP is unavailable -- the physics-group "
            "summary and comparative analysis both require real SHAP values (not a "
            "Gini-importance fallback, see the V6 FIX notes on create_physics_group_summary() "
            "and create_comparative_analysis()), so processing cannot proceed without them."
        )
    importance_df = create_comparative_analysis(shap_results, feature_names, OUTPUT_DIR)
    physics_df, physics_df_mean, physics_df_median, physics_group_counts = create_physics_group_summary(
        shap_results, feature_names, OUTPUT_DIR
    )
    create_shap_summary_plots(shap_results, feature_names, OUTPUT_DIR)

    # Save results
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    # Summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_labels_analyzed': len(results),
        'labels': list(results.keys()),
        'test_scores': {k: v['test_score'] for k, v in results.items()},
        'n_positive_samples': {k: v['n_positive'] for k, v in results.items()}
    }

    with open(OUTPUT_DIR / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save importance arrays
    importance_arrays = {
        fault_name: data['importance']
        for fault_name, data in results.items()
    }
    np.savez(OUTPUT_DIR / 'feature_importance.npz',
             feature_names=feature_names,
             **importance_arrays)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nPer-Label Analysis:")
    for fault_name, data in results.items():
        print(f"  {fault_name}:")
        print(f"    Positive samples: {data['n_positive']}")
        print(f"    Test accuracy: {data['test_score']:.3f}")

    print(f"\nPhysics Group Feature Counts:")
    print(physics_group_counts)

    print(f"\nPhysics Group Summary, real SHAP values (% of group-summed importance):")
    print(physics_df.to_string())

    print(f"\nPhysics Group Summary, real SHAP values (% of per-feature-MEAN importance -- controls for group size):")
    print(physics_df_mean.to_string())

    print(f"\nPhysics Group Summary, real SHAP values (% of per-feature-MEDIAN importance -- robust to a single outlier feature dominating a small group's mean):")
    print(physics_df_median.to_string())

    print(f"\nTop 10 Most Important Features (mean across all labels):")
    for i, (fname, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"  {i+1}. {fname}: {row['mean']:.4f} (std={row['std']:.4f})")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
