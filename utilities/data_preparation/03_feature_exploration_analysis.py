#!/usr/bin/env python3
"""
Feature Exploration Analysis - Python script version
Tests data loading and analysis BEFORE converting to notebook.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

print("="*60)
print("Feature Exploration Analysis")
print("="*60)

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

OUTPUT_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')

# ========================================
# Load engineered features
# ========================================
print("\n1. Loading engineered features...")

features_file = OUTPUT_DIR / 'features_engineered.pkl'

if not features_file.exists():
    print(f"❌ Features file not found: {features_file}")
    print("   Run the cluster pipeline first: ./run_pipeline.sh")
    exit(1)

with open(features_file, 'rb') as f:
    feature_data = pickle.load(f)

# Extract data
features_all = feature_data['features_all']
X_scaled = feature_data['X_scaled']
X_pca = feature_data['X_pca']
y_binary = feature_data.get('y_binary', None)
y_multilabel = feature_data.get('y_multilabel', None)
fault_column_names = feature_data.get('fault_column_names', [])

print(f"✓ Loaded features:")
print(f"  Full feature matrix: {features_all.shape}")
print(f"  Scaled features: {X_scaled.shape}")
print(f"  PCA components: {X_pca.shape}")
print(f"  Binary labels: {y_binary.shape if y_binary is not None else 'N/A'}")
print(f"  Multi-label: {y_multilabel.shape if y_multilabel is not None else 'N/A'}")
print(f"  Fault types: {fault_column_names}")

# ========================================
# 2. Feature statistics
# ========================================
print("\n2. Feature statistics...")

feature_cols = feature_data['feature_cols']
print(f"\nNumber of features: {len(feature_cols)}")

# Show sample features
print(f"\nFirst 10 features:")
for i, feat in enumerate(feature_cols[:10], 1):
    print(f"  {i:2d}. {feat}")

# Feature value ranges
print(f"\nScaled feature ranges:")
print(f"  Min: {X_scaled.min():.3f}")
print(f"  Max: {X_scaled.max():.3f}")
print(f"  Mean: {X_scaled.mean():.3f}")
print(f"  Std: {X_scaled.std():.3f}")

# ========================================
# 3. Label distribution
# ========================================
print("\n3. Label distribution...")

if y_binary is not None:
    print(f"\nBinary classification:")
    print(f"  Fault events: {y_binary.sum()}")
    print(f"  Normal events: {len(y_binary) - y_binary.sum()}")
    print(f"  Fault ratio: {y_binary.mean():.1%}")

if y_multilabel is not None and len(fault_column_names) > 0:
    print(f"\nMulti-label fault distribution:")
    for i, fault_name in enumerate(fault_column_names):
        count = y_multilabel[:, i].sum()
        print(f"  {fault_name}: {count} events")

# ========================================
# 4. PCA explained variance
# ========================================
print("\n4. PCA analysis...")

pca = feature_data['pca']
explained_var = pca.explained_variance_ratio_

print(f"\nPCA components: {len(explained_var)}")
print(f"Total explained variance: {explained_var.sum():.1%}")
print(f"\nTop 5 components:")
for i in range(min(5, len(explained_var))):
    print(f"  PC{i+1}: {explained_var[i]:.1%}")

# ========================================
# 5. Feature correlations (if visualization enabled)
# ========================================
print("\n5. Feature correlations...")

# Compute correlation matrix for a subset of features
n_features_to_show = min(20, len(feature_cols))
corr_matrix = np.corrcoef(X_scaled[:, :n_features_to_show].T)

print(f"\nCorrelation matrix computed for first {n_features_to_show} features")
print(f"  Shape: {corr_matrix.shape}")
print(f"  Max correlation (off-diagonal): {np.max(np.abs(corr_matrix - np.eye(n_features_to_show))):.3f}")

# ========================================
# 6. Summary
# ========================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"✓ Successfully loaded and analyzed features")
print(f"✓ Total events: {len(features_all)}")
print(f"✓ Total features: {len(feature_cols)}")
print(f"✓ PCA components: {X_pca.shape[1]}")
print(f"✓ Fault types: {len(fault_column_names)}")
print("="*60)

print("\n✅ Analysis complete - ready to convert to notebook!")
