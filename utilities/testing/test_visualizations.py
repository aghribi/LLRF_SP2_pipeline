#!/usr/bin/env python3
"""
Test script for visualization cells in 03_feature_exploration.ipynb
Verifies that all new visualization cells execute correctly.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*80)
print("TESTING VISUALIZATION CELLS")
print("="*80)

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
OUTPUT_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')

# Load data
features_file = OUTPUT_DIR / 'features_engineered.pkl'
print(f"\n1. Loading features from {features_file}...")
with open(features_file, 'rb') as f:
    data = pickle.load(f)

df_features_all = data['features_all']
X_scaled = data['X_scaled']
X_pca = data['X_pca']
y_binary = data['y_binary']
Y_multilabel = data['y_multilabel']
fault_column_names = data['fault_column_names']
scaler = data['scaler']
pca = data['pca']
df_metadata = data['metadata']
feature_cols = data['feature_cols']

print(f"   ✓ Loaded {len(df_features_all)} events with {len(feature_cols)} features")

# Compute feature_vars (needed for correlation cell)
feature_vars = X_scaled.var(axis=0)

# ============================================================================
# TEST 1: Feature Distribution Analysis
# ============================================================================
print(f"\n2. Testing Feature Distribution Analysis...")
try:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Feature variance histogram
    axes[0, 0].hist(feature_vars, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(feature_vars.mean(), color='red', linestyle='--', label=f'Mean: {feature_vars.mean():.3f}')
    axes[0, 0].axvline(np.median(feature_vars), color='orange', linestyle='--', label=f'Median: {np.median(feature_vars):.3f}')
    axes[0, 0].set_xlabel('Variance')
    axes[0, 0].set_ylabel('Number of Features')
    axes[0, 0].set_title('Feature Variance Distribution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Feature mean distribution
    feature_means = X_scaled.mean(axis=0)
    axes[0, 1].hist(feature_means, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', label='Zero (expected after scaling)')
    axes[0, 1].set_xlabel('Mean Value')
    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].set_title('Feature Mean Distribution (Should be ~0 after scaling)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Feature sparsity
    sparsity = (df_features_all == 0).sum(axis=0) / len(df_features_all) * 100
    axes[1, 0].hist(sparsity, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Sparsity (%)')
    axes[1, 0].set_ylabel('Number of Features')
    axes[1, 0].set_title('Feature Sparsity Distribution', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Feature std distribution
    feature_stds = X_scaled.std(axis=0)
    axes[1, 1].hist(feature_stds, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(1.0, color='red', linestyle='--', label='1.0 (expected after scaling)')
    axes[1, 1].set_xlabel('Standard Deviation')
    axes[1, 1].set_ylabel('Number of Features')
    axes[1, 1].set_title('Feature Std Distribution (Should be ~1 after scaling)', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'test_feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Feature distribution plots created successfully")
except Exception as e:
    print(f"   ✗ ERROR in feature distribution: {e}")
    raise

# ============================================================================
# TEST 2: Correlation Analysis
# ============================================================================
print(f"\n3. Testing Correlation Analysis...")
try:
    n_features_to_analyze = min(100, len(feature_cols))
    top_var_indices = np.argsort(feature_vars)[-n_features_to_analyze:]
    X_sample = X_scaled[:, top_var_indices]
    feature_cols_sample = [feature_cols[i] for i in top_var_indices]

    corr_matrix = np.corrcoef(X_sample.T)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 1. Correlation heatmap
    im = axes[0].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[0].set_title(f'Feature Correlation Matrix (Top {n_features_to_analyze} by Variance)', fontweight='bold')
    axes[0].set_xlabel('Feature Index')
    axes[0].set_ylabel('Feature Index')
    plt.colorbar(im, ax=axes[0], label='Correlation')

    # 2. Correlation distribution
    corr_values = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    axes[1].hist(corr_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', label='Zero correlation')
    axes[1].set_xlabel('Correlation Coefficient')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Pairwise Feature Correlation Distribution', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'test_feature_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Correlation plots created successfully")
except Exception as e:
    print(f"   ✗ ERROR in correlation analysis: {e}")
    raise

# ============================================================================
# TEST 3: Enhanced PCA Visualization
# ============================================================================
print(f"\n4. Testing Enhanced PCA Visualization...")
try:
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. PC1 vs PC2 colored by fault/normal
    colors = ['steelblue' if y == 0 else 'coral' for y in y_binary]
    axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=30)
    axes[0, 0].set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% var)')
    axes[0, 0].set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% var)')
    axes[0, 0].set_title('PCA: Fault vs Normal Events (PC1 vs PC2)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    legend_elements = [
        Patch(facecolor='coral', label=f'Fault ({y_binary.sum()} events)'),
        Patch(facecolor='steelblue', label=f'Normal ({(y_binary==0).sum()} events)')
    ]
    axes[0, 0].legend(handles=legend_elements, loc='upper right')

    # 2. PC2 vs PC3
    axes[0, 1].scatter(X_pca[:, 1], X_pca[:, 2], c=colors, alpha=0.6, s=30)
    axes[0, 1].set_xlabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% var)')
    axes[0, 1].set_ylabel(f'PC3 ({100*pca.explained_variance_ratio_[2]:.1f}% var)')
    axes[0, 1].set_title('PCA: PC2 vs PC3', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. PC1 vs PC3
    axes[0, 2].scatter(X_pca[:, 0], X_pca[:, 2], c=colors, alpha=0.6, s=30)
    axes[0, 2].set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% var)')
    axes[0, 2].set_ylabel(f'PC3 ({100*pca.explained_variance_ratio_[2]:.1f}% var)')
    axes[0, 2].set_title('PCA: PC1 vs PC3', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Explained variance per component
    axes[1, 0].bar(range(1, min(21, len(pca.explained_variance_ratio_)+1)),
                   pca.explained_variance_ratio_[:20], color='steelblue', alpha=0.7)
    axes[1, 0].set_xlabel('Principal Component')
    axes[1, 0].set_ylabel('Explained Variance Ratio')
    axes[1, 0].set_title('Explained Variance by Component (Top 20)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 5. Cumulative explained variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    axes[1, 1].plot(range(1, len(cumvar)+1), cumvar, marker='o', linewidth=2, markersize=3)
    axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='95% variance', linewidth=2)
    axes[1, 1].axhline(y=0.99, color='orange', linestyle='--', label='99% variance', linewidth=2)
    axes[1, 1].set_xlabel('Number of Components')
    axes[1, 1].set_ylabel('Cumulative Explained Variance')
    axes[1, 1].set_title('PCA Cumulative Explained Variance', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    # 6. PC1 histogram by class
    pc1_fault = X_pca[y_binary == 1, 0]
    pc1_normal = X_pca[y_binary == 0, 0]
    axes[1, 2].hist(pc1_normal, bins=30, alpha=0.6, label='Normal', color='steelblue', edgecolor='black')
    axes[1, 2].hist(pc1_fault, bins=30, alpha=0.6, label='Fault', color='coral', edgecolor='black')
    axes[1, 2].set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% var)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('PC1 Distribution: Fault vs Normal', fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'test_pca_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Enhanced PCA plots created successfully")
except Exception as e:
    print(f"   ✗ ERROR in PCA visualization: {e}")
    raise

# ============================================================================
# TEST 4: Discriminative Features
# ============================================================================
print(f"\n5. Testing Discriminative Features Visualization...")
try:
    # Get faults with enough samples
    faults_to_analyze = []
    for fault_idx, fault_name in enumerate(fault_column_names):
        fault_mask = Y_multilabel[:, fault_idx] == 1
        if fault_mask.sum() >= 5:
            faults_to_analyze.append((fault_idx, fault_name, fault_mask.sum()))

    n_faults = len(faults_to_analyze)
    if n_faults == 0:
        print(f"   ⚠ No faults with >= 5 samples, skipping discriminative features test")
    else:
        fig, axes = plt.subplots(n_faults, 2, figsize=(18, 5*n_faults))
        if n_faults == 1:
            axes = axes.reshape(1, -1)

        for plot_idx, (fault_idx, fault_name, n_events) in enumerate(faults_to_analyze):
            fault_mask = Y_multilabel[:, fault_idx] == 1
            normal_mask = y_binary == 0

            # Compute mean difference
            fault_features = X_scaled[fault_mask].mean(axis=0)
            normal_features = X_scaled[normal_mask].mean(axis=0)
            diff = np.abs(fault_features - normal_features)

            # Top 15 discriminative features
            top_indices = np.argsort(diff)[-15:][::-1]
            top_features = [feature_cols[i] for i in top_indices]
            top_diff = diff[top_indices]

            # Plot 1: Bar chart
            y_pos = np.arange(len(top_features))
            axes[plot_idx, 0].barh(y_pos, top_diff, color='steelblue', alpha=0.7)
            axes[plot_idx, 0].set_yticks(y_pos)
            axes[plot_idx, 0].set_yticklabels([f[:35] for f in top_features], fontsize=9)
            axes[plot_idx, 0].set_xlabel('Mean Absolute Difference (Fault - Normal)')
            axes[plot_idx, 0].set_title(f'{fault_name}\nTop 15 Discriminative Features ({n_events} events)',
                                        fontweight='bold')
            axes[plot_idx, 0].grid(True, alpha=0.3, axis='x')
            axes[plot_idx, 0].invert_yaxis()

            # Plot 2: Distribution of top feature
            top_feat_idx = top_indices[0]
            top_feat_name = feature_cols[top_feat_idx][:45]

            normal_vals = X_scaled[normal_mask, top_feat_idx]
            fault_vals = X_scaled[fault_mask, top_feat_idx]

            axes[plot_idx, 1].hist(normal_vals, bins=30, alpha=0.6, label=f'Normal (n={normal_mask.sum()})',
                                  color='steelblue', edgecolor='black')
            axes[plot_idx, 1].hist(fault_vals, bins=30, alpha=0.6, label=f'Fault (n={fault_mask.sum()})',
                                  color='coral', edgecolor='black')
            axes[plot_idx, 1].set_xlabel('Feature Value (scaled)')
            axes[plot_idx, 1].set_ylabel('Frequency')
            axes[plot_idx, 1].set_title(f'Distribution: {top_feat_name}\n(Most discriminative feature)',
                                       fontweight='bold')
            axes[plot_idx, 1].legend()
            axes[plot_idx, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'test_discriminative_features.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Discriminative features plots created successfully")
except Exception as e:
    print(f"   ✗ ERROR in discriminative features: {e}")
    raise

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print("ALL VISUALIZATION TESTS PASSED ✓")
print(f"{'='*80}")
print(f"\nGenerated test plots:")
print(f"  1. {OUTPUT_DIR / 'test_feature_distributions.png'}")
print(f"  2. {OUTPUT_DIR / 'test_feature_correlations.png'}")
print(f"  3. {OUTPUT_DIR / 'test_pca_detailed.png'}")
print(f"  4. {OUTPUT_DIR / 'test_discriminative_features.png'}")
print(f"\n✅ All visualization cells are working correctly!")
print(f"{'='*80}")
