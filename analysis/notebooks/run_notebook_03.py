#!/usr/bin/env python
"""
Execute notebook 03 cells: Comprehensive Feature Engineering & Exploration
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

print("="*70)
print("NOTEBOOK 03: COMPREHENSIVE FEATURE EXPLORATION")
print("="*70)
print("\n✓ Imports complete\n")

# Define data path
DATA_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')
FEATURES_FILE = DATA_DIR / 'features_engineered.pkl'
OUTPUT_DIR = DATA_DIR

print(f"Loading data from: {FEATURES_FILE}")
print(f"File size: {FEATURES_FILE.stat().st_size / (1024**3):.2f} GB")

# Load data
print("Loading pickle file...")
with open(FEATURES_FILE, 'rb') as f:
    data = pickle.load(f)

# Extract components
df_features = data['features_all']
feature_cols = data['feature_cols']
X_scaled = data['X_scaled']
X_pca = data['X_pca']
y_binary = data['y_binary']
Y_multilabel = data['y_multilabel']
fault_names = data['fault_column_names']
metadata = data['metadata']
scaler = data['scaler']
pca = data['pca']

# Convert metadata to DataFrame if needed
if isinstance(metadata, list):
    df_metadata = pd.DataFrame(metadata)
else:
    df_metadata = metadata

print(f"\n✓ Dataset loaded successfully")
print(f"  Total events: {len(df_features):,}")
print(f"  Features: {len(feature_cols):,}")
print(f"  PCA components: {X_pca.shape[1]}")
print(f"  Fault types: {len(fault_names)}")
print(f"  Fault events: {y_binary.sum():,} ({100*y_binary.sum()/len(y_binary):.1f}%)")
print(f"  Normal events: {(y_binary==0).sum():,} ({100*(y_binary==0).sum()/len(y_binary):.1f}%)")

# Feature taxonomy analysis
print("\n" + "="*70)
print("FEATURE TAXONOMY")
print("="*70)

# Categorize features by type
feature_categories = {
    'Statistical - Location': ['_mean', '_median', '_min', '_max'],
    'Statistical - Dispersion': ['_std', '_iqr', '_range'],
    'Statistical - Shape': ['_skewness', '_kurtosis'],
    'Statistical - Energy': ['_energy', '_rms'],
    'Statistical - Events': ['_n_peaks', '_n_zero_crossings'],
    'Statistical - Correlation': ['_autocorr'],
    'Temporal - Trends': ['_slope', '_accel'],
    'Temporal - Variance': ['_var_ratio', 'ratio'],
    'Change Point': ['cusum'],
    'Physics-Based': ['ql', 'detuning', 'control', 'phase_jitter', 'phase_excursion'],
}

feature_counts = {}
for category, keywords in feature_categories.items():
    count = sum(1 for f in feature_cols if any(kw in f for kw in keywords))
    feature_counts[category] = count
    print(f"  {category:30s}: {count:4d} features")

print(f"\n  Total categorized: {sum(feature_counts.values())}/{len(feature_cols)}")
print(f"  Uncategorized: {len(feature_cols) - sum(feature_counts.values())}")

# Overall feature statistics
print("\n" + "="*70)
print("FEATURE DISTRIBUTION ANALYSIS")
print("="*70)

feature_vars = X_scaled.var(axis=0)
feature_means = X_scaled.mean(axis=0)
feature_stds = X_scaled.std(axis=0)
sparsity = (df_features == 0).sum(axis=0) / len(df_features) * 100

print(f"\nFeature variance statistics:")
print(f"  Mean: {feature_vars.mean():.3f}")
print(f"  Median: {np.median(feature_vars):.3f}")
print(f"  High variance (>1.5): {(feature_vars > 1.5).sum()}")
print(f"  Low variance (<0.1): {(feature_vars < 0.1).sum()}")

print(f"\nFeature sparsity statistics:")
print(f"  Mean sparsity: {sparsity.mean():.1f}%")
print(f"  Median sparsity: {np.median(sparsity):.1f}%")
print(f"  Very sparse (>50% zeros): {(sparsity > 50).sum()}")

# Visualization 1: Feature distributions
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Variance
axes[0, 0].hist(feature_vars, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(feature_vars.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {feature_vars.mean():.3f}')
axes[0, 0].axvline(np.median(feature_vars), color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(feature_vars):.3f}')
axes[0, 0].set_xlabel('Variance', fontsize=11)
axes[0, 0].set_ylabel('Number of Features', fontsize=11)
axes[0, 0].set_title('Feature Variance Distribution', fontweight='bold', fontsize=13)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Mean (should be ~0 after scaling)
axes[0, 1].hist(feature_means, bins=50, color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (expected)')
axes[0, 1].set_xlabel('Mean Value', fontsize=11)
axes[0, 1].set_ylabel('Number of Features', fontsize=11)
axes[0, 1].set_title('Feature Mean Distribution (Should be ~0)', fontweight='bold', fontsize=13)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Sparsity
axes[1, 0].hist(sparsity, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Sparsity (%)', fontsize=11)
axes[1, 0].set_ylabel('Number of Features', fontsize=11)
axes[1, 0].set_title('Feature Sparsity Distribution', fontweight='bold', fontsize=13)
axes[1, 0].grid(True, alpha=0.3)

# Std (should be ~1 after scaling)
axes[1, 1].hist(feature_stds, bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(1.0, color='red', linestyle='--', linewidth=2, label='1.0 (expected)')
axes[1, 1].set_xlabel('Standard Deviation', fontsize=11)
axes[1, 1].set_ylabel('Number of Features', fontsize=11)
axes[1, 1].set_title('Feature Std Distribution (Should be ~1)', fontweight='bold', fontsize=13)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_file = OUTPUT_DIR / '03_feature_distributions.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: {output_file}")

# PCA Analysis
print("\n" + "="*70)
print("PCA DIMENSIONALITY REDUCTION ANALYSIS")
print("="*70)

cumvar = np.cumsum(pca.explained_variance_ratio_)

def components_for_variance(cumvar, threshold):
    idx = np.argmax(cumvar >= threshold)
    if cumvar[idx] >= threshold:
        return idx + 1
    else:
        return f"Not reached (max: {cumvar[-1]:.1%})"

print(f"\nPCA Results:")
print(f"  Total components: {X_pca.shape[1]} ({cumvar[-1]:.1%} variance)")
print(f"  For 90% variance: {components_for_variance(cumvar, 0.90)} components")
print(f"  For 95% variance: {components_for_variance(cumvar, 0.95)} components")
print(f"  For 99% variance: {components_for_variance(cumvar, 0.99)}")

# Visualization 2: PCA detailed analysis
from matplotlib.patches import Patch

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

colors = ['steelblue' if y == 0 else 'coral' for y in y_binary]

# PC1 vs PC2
axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=20)
axes[0, 0].set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% var)', fontsize=10)
axes[0, 0].set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% var)', fontsize=10)
axes[0, 0].set_title('PC1 vs PC2', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(handles=[Patch(facecolor='coral', label='Fault'),
                          Patch(facecolor='steelblue', label='Normal')])

# PC2 vs PC3
axes[0, 1].scatter(X_pca[:, 1], X_pca[:, 2], c=colors, alpha=0.6, s=20)
axes[0, 1].set_xlabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% var)', fontsize=10)
axes[0, 1].set_ylabel(f'PC3 ({100*pca.explained_variance_ratio_[2]:.1f}% var)', fontsize=10)
axes[0, 1].set_title('PC2 vs PC3', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# PC1 vs PC3
axes[0, 2].scatter(X_pca[:, 0], X_pca[:, 2], c=colors, alpha=0.6, s=20)
axes[0, 2].set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% var)', fontsize=10)
axes[0, 2].set_ylabel(f'PC3 ({100*pca.explained_variance_ratio_[2]:.1f}% var)', fontsize=10)
axes[0, 2].set_title('PC1 vs PC3', fontweight='bold', fontsize=12)
axes[0, 2].grid(True, alpha=0.3)

# Explained variance
axes[1, 0].bar(range(1, 21), pca.explained_variance_ratio_[:20], color='steelblue', alpha=0.7)
axes[1, 0].set_xlabel('Component', fontsize=10)
axes[1, 0].set_ylabel('Explained Variance', fontsize=10)
axes[1, 0].set_title('Explained Variance (Top 20)', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Cumulative variance
axes[1, 1].plot(range(1, len(cumvar)+1), cumvar, marker='o', linewidth=2, markersize=2)
axes[1, 1].axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95%')
axes[1, 1].axhline(y=0.99, color='orange', linestyle='--', linewidth=2, label='99%')
axes[1, 1].set_xlabel('Components', fontsize=10)
axes[1, 1].set_ylabel('Cumulative Variance', fontsize=10)
axes[1, 1].set_title('Cumulative Explained Variance', fontweight='bold', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# PC1 histogram by class
axes[1, 2].hist(X_pca[y_binary==0, 0], bins=40, alpha=0.6, label='Normal',
               color='steelblue', edgecolor='black')
axes[1, 2].hist(X_pca[y_binary==1, 0], bins=40, alpha=0.6, label='Fault',
               color='coral', edgecolor='black')
axes[1, 2].set_xlabel('PC1', fontsize=10)
axes[1, 2].set_ylabel('Frequency', fontsize=10)
axes[1, 2].set_title('PC1 Distribution by Class', fontweight='bold', fontsize=12)
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_file = OUTPUT_DIR / '03_pca_detailed.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_file}")

# Discriminative features per fault type
print("\n" + "="*70)
print("DISCRIMINATIVE FEATURES ANALYSIS")
print("="*70)

faults = [(i, name, (Y_multilabel[:, i] == 1).sum())
          for i, name in enumerate(fault_names) if (Y_multilabel[:, i] == 1).sum() >= 10]

print(f"\nAnalyzing {len(faults)} fault types with ≥10 events:")

n_faults = len(faults)
fig, axes = plt.subplots(n_faults, 2, figsize=(18, 5*n_faults))
if n_faults == 1:
    axes = axes.reshape(1, -1)

for idx, (fault_idx, fault_name, n_events) in enumerate(faults):
    fault_mask = Y_multilabel[:, fault_idx] == 1
    normal_mask = y_binary == 0

    diff = np.abs(X_scaled[fault_mask].mean(axis=0) - X_scaled[normal_mask].mean(axis=0))
    top_idx = np.argsort(diff)[-15:][::-1]

    print(f"\n  {fault_name} ({n_events} events):")
    print(f"    Top 5 discriminative features:")
    for rank, i in enumerate(top_idx[:5], 1):
        print(f"      {rank}. {feature_cols[i][:40]} (Δ={diff[i]:.3f})")

    # Bar chart
    axes[idx, 0].barh(range(15), diff[top_idx], color='steelblue', alpha=0.7)
    axes[idx, 0].set_yticks(range(15))
    axes[idx, 0].set_yticklabels([feature_cols[i][:35] for i in top_idx], fontsize=8)
    axes[idx, 0].set_xlabel('Mean Absolute Difference', fontsize=10)
    axes[idx, 0].set_title(f'{fault_name}\nTop 15 Features ({n_events} events)',
                          fontweight='bold', fontsize=11)
    axes[idx, 0].invert_yaxis()
    axes[idx, 0].grid(True, alpha=0.3, axis='x')

    # Histogram
    axes[idx, 1].hist(X_scaled[normal_mask, top_idx[0]], bins=30, alpha=0.6,
                     label='Normal', color='steelblue', edgecolor='black')
    axes[idx, 1].hist(X_scaled[fault_mask, top_idx[0]], bins=30, alpha=0.6,
                     label='Fault', color='coral', edgecolor='black')
    axes[idx, 1].set_xlabel('Value (scaled)', fontsize=10)
    axes[idx, 1].set_ylabel('Frequency', fontsize=10)
    axes[idx, 1].set_title(f'{feature_cols[top_idx[0]][:40]}', fontsize=9)
    axes[idx, 1].legend()
    axes[idx, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_file = OUTPUT_DIR / '03_discriminative_features.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: {output_file}")

# Feature correlation analysis
print("\n" + "="*70)
print("FEATURE CORRELATION ANALYSIS")
print("="*70)

# Compute correlation matrix for a subset of features (too large for all)
sample_features = np.random.choice(len(feature_cols), min(100, len(feature_cols)), replace=False)
sample_feature_names = [feature_cols[i] for i in sample_features]
X_sample = X_scaled[:, sample_features]

corr_matrix = np.corrcoef(X_sample.T)

print(f"\nCorrelation statistics (sample of {len(sample_features)} features):")
corr_values = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
print(f"  Mean absolute correlation: {np.abs(corr_values).mean():.3f}")
print(f"  High correlation (|r|>0.9): {(np.abs(corr_values) > 0.9).sum()} pairs")
print(f"  Very high correlation (|r|>0.95): {(np.abs(corr_values) > 0.95).sum()} pairs")

print("\n" + "="*70)
print("NOTEBOOK 03 COMPLETE")
print("="*70)
print(f"\nKey Results:")
print(f"  - Analyzed {len(feature_cols):,} features across {len(df_features):,} events")
print(f"  - Feature taxonomy: {len(feature_categories)} categories")
print(f"  - PCA: {X_pca.shape[1]} components capture {100*cumvar[-1]:.1f}% variance")
print(f"  - Identified discriminative features for {len(faults)} fault types")
print(f"  - All visualizations saved to {OUTPUT_DIR}")
print("\n✓ All outputs generated successfully")
