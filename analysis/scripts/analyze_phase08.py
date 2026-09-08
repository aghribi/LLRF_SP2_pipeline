#!/usr/bin/env python
"""
Phase 08 Root Cause Analysis - Comprehensive Report
Loads pre-computed results and generates analysis with visualizations
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')
RESULTS_FILE = OUTPUT_DIR / 'step_08_phase3a' / 'root_cause.pkl'
FIGURES_DIR = Path('../outputs')
FIGURES_DIR.mkdir(exist_ok=True)

# Visualization setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("PHASE 08: ROOT CAUSE IDENTIFICATION - COMPREHENSIVE ANALYSIS")
print("="*70)
print()

# Load results
print(f"Loading results from: {RESULTS_FILE}")
with open(RESULTS_FILE, 'rb') as f:
    results = pickle.load(f)

# Extract components
model = results['model']
y_pred = results['predictions']
data_splits = results['data_splits']
fault_names = results['fault_names']

X_test = data_splits['X_test']
y_test = data_splits['y_test']

# Generate probabilities from model
y_prob = model.predict_proba(X_test)

print(f"✓ Results loaded")
print(f"\nDataset Summary:")
print(f"  Test events: {len(y_test)}")
print(f"  Features: {X_test.shape[1]}")
print(f"  Fault types: {len(fault_names)}")
print(f"  Fault names: {fault_names}")
print()

#========================================
# 1. OVERALL PERFORMANCE
#========================================
print("="*70)
print("1. OVERALL PERFORMANCE METRICS")
print("="*70)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='weighted', zero_division=0
)

print(f"\nModel: Random Forest")
print(f"Test Events: {len(y_test)}")
print(f"")
print(f"Overall Performance:")
print(f"  Accuracy:        {accuracy:.3f}")
print(f"  Precision (wtd): {precision:.3f}")
print(f"  Recall (wtd):    {recall:.3f}")
print(f"  F1-Score (wtd):  {f1:.3f}")
print()
print(f"⚠️  Note: Performance relative to 'first active fault' pseudo-labels")
print()

#========================================
# 2. PER-CLASS PERFORMANCE
#========================================
print("="*70)
print("2. PER-CLASS PERFORMANCE")
print("="*70)

per_class_prec, per_class_rec, per_class_f1, per_class_support = precision_recall_fscore_support(
    y_test, y_pred, labels=range(len(fault_names)), zero_division=0
)

per_class_df = pd.DataFrame({
    'Fault_Type': fault_names,
    'Precision': per_class_prec,
    'Recall': per_class_rec,
    'F1_Score': per_class_f1,
    'Support': per_class_support,
})
per_class_df['Percentage'] = 100 * per_class_df['Support'] / per_class_df['Support'].sum()

print("\nPer-Class Metrics:")
print(per_class_df.to_string(index=False))
print()

# Visualize per-class performance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# F1-Score
axes[0].bar(range(len(fault_names)), per_class_f1, color='steelblue', edgecolor='black')
axes[0].set_xticks(range(len(fault_names)))
axes[0].set_xticklabels(fault_names, rotation=45, ha='right', fontsize=9)
axes[0].set_ylabel('F1-Score', fontsize=12)
axes[0].set_ylim([0, 1])
axes[0].set_title('Per-Class F1-Score (Root Cause Identification)', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Support distribution
axes[1].bar(range(len(fault_names)), per_class_support, color='coral', edgecolor='black')
axes[1].set_xticks(range(len(fault_names)))
axes[1].set_xticklabels(fault_names, rotation=45, ha='right', fontsize=9)
axes[1].set_ylabel('Number of Events', fontsize=12)
axes[1].set_title('Root Cause Distribution (Test Set)', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'phase08_per_class_performance.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Per-class performance plots saved")

#========================================
# 3. CONFUSION MATRIX
#========================================
print("\n" + "="*70)
print("3. CONFUSION MATRIX ANALYSIS")
print("="*70)

cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=fault_names, yticklabels=fault_names,
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_ylabel('True Root Cause', fontsize=12)
axes[0].set_xlabel('Predicted Root Cause', fontsize=12)
axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='both', labelsize=9)

# Normalized
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=fault_names, yticklabels=fault_names,
            ax=axes[1], vmin=0, vmax=1, cbar_kws={'label': 'Recall'})
axes[1].set_ylabel('True Root Cause', fontsize=12)
axes[1].set_xlabel('Predicted Root Cause', fontsize=12)
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='both', labelsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'phase08_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Confusion matrix saved")

#========================================
# 4. FEATURE IMPORTANCE
#========================================
print("\n" + "="*70)
print("4. FEATURE IMPORTANCE ANALYSIS")
print("="*70)

feature_importances = model.feature_importances_
feature_names_list = [f'Feature_{i}' for i in range(len(feature_importances))]

# Sort by importance
importance_df = pd.DataFrame({
    'Feature': feature_names_list,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

top_n = 30
top_features = importance_df.head(top_n)

print(f"\nTop {top_n} Most Important Features:")
print(top_features.head(15).to_string(index=False))
print("  ...")
print()

# Visualize
plt.figure(figsize=(12, 10))
plt.barh(range(top_n), top_features['Importance'].values, color='steelblue')
plt.yticks(range(top_n), top_features['Feature'].values, fontsize=9)
plt.xlabel('Feature Importance (Gini)', fontsize=12)
plt.title(f'Top {top_n} Features for Root Cause Identification', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'phase08_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Feature importance plot saved")

#========================================
# 5. PREDICTION CONFIDENCE
#========================================
print("\n" + "="*70)
print("5. PREDICTION CONFIDENCE ANALYSIS")
print("="*70)

max_probs = y_prob.max(axis=1)
correct = (y_pred == y_test)

print(f"\nConfidence Statistics:")
print(f"  Mean:   {max_probs.mean():.3f}")
print(f"  Median: {np.median(max_probs):.3f}")
print(f"  Min:    {max_probs.min():.3f}")
print(f"  Max:    {max_probs.max():.3f}")

# Confidence bins
bins = [0, 0.50, 0.70, 0.85, 1.0]
labels = ['Low (0-50%)', 'Medium (50-70%)', 'High (70-85%)', 'Very High (85-100%)']
confidence_bins = pd.cut(max_probs, bins=bins, labels=labels)

print("\nConfidence Distribution:")
for label in labels:
    mask = (confidence_bins == label)
    count = mask.sum()
    acc_in_bin = correct[mask].mean() if count > 0 else 0
    print(f"  {label:25s}: {count:4d} events ({count/len(y_test)*100:5.1f}%), Accuracy: {acc_in_bin:.3f}")
print()

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
axes[0].hist(max_probs, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(max_probs.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {max_probs.mean():.3f}')
axes[0].set_xlabel('Max Probability', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distribution of Prediction Confidence', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Confidence vs Correctness
axes[1].scatter(max_probs[correct], np.ones(correct.sum()),
                alpha=0.3, color='green', label='Correct', s=20)
axes[1].scatter(max_probs[~correct], np.zeros((~correct).sum()),
                alpha=0.3, color='red', label='Incorrect', s=20)
axes[1].set_xlabel('Max Probability', fontsize=12)
axes[1].set_ylabel('Correctness', fontsize=12)
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(['Incorrect', 'Correct'])
axes[1].set_title('Confidence vs Correctness', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'phase08_confidence_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Confidence analysis plots saved")

#========================================
# SUMMARY
#========================================
print("\n" + "="*70)
print("SUMMARY AND CONCLUSIONS")
print("="*70)

print(f"\n**Overall Performance**:")
print(f"  - Accuracy: {accuracy:.1%}")
print(f"  - Weighted F1-score: {f1:.3f}")
print(f"  - Test events: {len(y_test)} (multi-fault events from Phase 07)")

print(f"\n**Best Performing Root Causes** (F1 > 0.70):")
for idx, row in per_class_df[per_class_df['F1_Score'] > 0.70].iterrows():
    print(f"  - {row['Fault_Type']}: F1={row['F1_Score']:.3f}, Support={int(row['Support'])}")

print(f"\n**Challenging Root Causes** (F1 < 0.50):")
for idx, row in per_class_df[per_class_df['F1_Score'] < 0.50].iterrows():
    print(f"  - {row['Fault_Type']}: F1={row['F1_Score']:.3f}, Support={int(row['Support'])}")

print(f"\n**Confidence**:")
high_conf = (max_probs > 0.85).sum()
low_conf = (max_probs < 0.70).sum()
print(f"  - High confidence predictions (>85%): {high_conf} ({high_conf/len(y_test)*100:.1f}%)")
print(f"  - Low confidence (needs review) (<70%): {low_conf} ({low_conf/len(y_test)*100:.1f}%)")

print(f"\n**Key Limitations**:")
print(f"  - Pseudo-labels: 'First active fault' heuristic, not ground truth")
print(f"  - No temporal onset analysis included")
print(f"  - Physics rules not validated")

print(f"\n**Recommendations**:")
print(f"  1. Deploy ML for high-F1 root causes with confidence >85%")
print(f"  2. Use temporal onset detection for ambiguous cases")
print(f"  3. Apply physics rules for low-support fault types")
print(f"  4. Flag low-confidence predictions for expert review")

print("\n" + "="*70)
print("✓ Phase 08 analysis complete!")
print(f"✓ All figures saved to: {FIGURES_DIR}")
print("="*70)
