#!/usr/bin/env python
"""
Phase 09 Fault Subtype Clustering - Comprehensive Report
Loads pre-computed clustering results and generates analysis
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')
RESULTS_FILE = OUTPUT_DIR / 'step_09_phase3b' / 'subtype_clustering.pkl'
FIGURES_DIR = Path('../outputs')
FIGURES_DIR.mkdir(exist_ok=True)

# Visualization setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("PHASE 09: FAULT SUBTYPE CLUSTERING - COMPREHENSIVE ANALYSIS")
print("="*70)
print()

# Load results
print(f"Loading results from: {RESULTS_FILE}")
with open(RESULTS_FILE, 'rb') as f:
    results = pickle.load(f)

fault_types = list(results.keys())
print(f"✓ Results loaded for {len(fault_types)} fault types")
print(f"  Fault types: {fault_types}")
print()

#========================================
# 1. CLUSTERING SUMMARY
#========================================
print("="*70)
print("1. CLUSTERING SUMMARY - SUB-TYPES PER FAULT")
print("="*70)
print()

summary_data = []
for fault_name in fault_types:
    fault_result = results[fault_name]
    n_clusters = fault_result['n_clusters']
    silhouette = fault_result['silhouette']
    labels = fault_result['labels']
    n_events = len(labels)

    summary_data.append({
        'Fault_Type': fault_name,
        'Events': n_events,
        'Subtypes': n_clusters,
        'Silhouette': silhouette
    })

summary_df = pd.DataFrame(summary_data)
print("Clustering Results:")
print(summary_df.to_string(index=False))
print()

# Interpretation
print("Silhouette Score Interpretation:")
print("  > 0.70: Strong, well-separated clusters")
print("  0.50-0.70: Reasonable cluster structure")
print("  0.25-0.50: Weak, overlapping clusters")
print("  < 0.25: No meaningful cluster structure")
print()

#========================================
# 2. VISUALIZE SUBTYPE DISTRIBUTION
#========================================
print("="*70)
print("2. SUBTYPE DISTRIBUTION PER FAULT")
print("="*70)
print()

# Calculate distribution for each fault
n_faults = len(fault_types)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, fault_name in enumerate(fault_types):
    fault_result = results[fault_name]
    labels = fault_result['labels']
    n_clusters = fault_result['n_clusters']
    silhouette = fault_result['silhouette']

    # Count events per cluster
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Plot
    ax = axes[idx]
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    bars = ax.bar(unique_labels, counts, color=colors, edgecolor='black')
    ax.set_xlabel('Subtype ID', fontsize=10)
    ax.set_ylabel('Events', fontsize=10)
    ax.set_title(f'{fault_name}\n({n_clusters} subtypes, Silh={silhouette:.2f})',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add percentages
    total = len(labels)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count/total*100:.0f}%',
                ha='center', va='bottom', fontsize=8)

    print(f"{fault_name}:")
    for label, count in zip(unique_labels, counts):
        print(f"  Subtype {label}: {count} events ({count/total*100:.1f}%)")
    print()

# Remove extra subplot
if n_faults < len(axes):
    fig.delaxes(axes[n_faults])

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'phase09_subtype_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Subtype distribution plot saved")

#========================================
# 3. SILHOUETTE SCORE ANALYSIS
#========================================
print("\n" + "="*70)
print("3. CLUSTERING QUALITY ANALYSIS")
print("="*70)
print()

# Categorize by quality
strong = summary_df[summary_df['Silhouette'] > 0.70]
reasonable = summary_df[(summary_df['Silhouette'] >= 0.50) & (summary_df['Silhouette'] <= 0.70)]
weak = summary_df[(summary_df['Silhouette'] >= 0.25) & (summary_df['Silhouette'] < 0.50)]
poor = summary_df[summary_df['Silhouette'] < 0.25]

print(f"Strong Clustering (Silhouette > 0.70): {len(strong)} fault types")
if len(strong) > 0:
    for _, row in strong.iterrows():
        print(f"  - {row['Fault_Type']}: {row['Subtypes']} subtypes, Silh={row['Silhouette']:.3f}")
print()

print(f"Reasonable Clustering (0.50-0.70): {len(reasonable)} fault types")
if len(reasonable) > 0:
    for _, row in reasonable.iterrows():
        print(f"  - {row['Fault_Type']}: {row['Subtypes']} subtypes, Silh={row['Silhouette']:.3f}")
print()

print(f"Weak Clustering (0.25-0.50): {len(weak)} fault types")
if len(weak) > 0:
    for _, row in weak.iterrows():
        print(f"  - {row['Fault_Type']}: {row['Subtypes']} subtypes, Silh={row['Silhouette']:.3f}")
print()

print(f"Poor Clustering (< 0.25): {len(poor)} fault types")
if len(poor) > 0:
    for _, row in poor.iterrows():
        print(f"  - {row['Fault_Type']}: {row['Subtypes']} subtypes, Silh={row['Silhouette']:.3f}")
print()

# Visualize silhouette scores
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot
colors = ['green' if s > 0.70 else 'orange' if s > 0.50 else 'coral' if s > 0.25 else 'red'
          for s in summary_df['Silhouette']]
axes[0].barh(range(len(fault_types)), summary_df['Silhouette'], color=colors, edgecolor='black')
axes[0].set_yticks(range(len(fault_types)))
axes[0].set_yticklabels(summary_df['Fault_Type'], fontsize=9)
axes[0].set_xlabel('Silhouette Score', fontsize=12)
axes[0].set_title('Clustering Quality by Fault Type', fontsize=14, fontweight='bold')
axes[0].axvline(0.70, color='green', linestyle='--', alpha=0.5, label='Strong (>0.70)')
axes[0].axvline(0.50, color='orange', linestyle='--', alpha=0.5, label='Reasonable (>0.50)')
axes[0].axvline(0.25, color='red', linestyle='--', alpha=0.5, label='Weak (>0.25)')
axes[0].legend(fontsize=9)
axes[0].grid(axis='x', alpha=0.3)

# Number of subtypes vs Silhouette
scatter_colors = ['green' if s > 0.70 else 'orange' if s > 0.50 else 'coral' if s > 0.25 else 'red'
                  for s in summary_df['Silhouette']]
axes[1].scatter(summary_df['Subtypes'], summary_df['Silhouette'],
                c=scatter_colors, s=200, alpha=0.6, edgecolor='black', linewidth=2)
for _, row in summary_df.iterrows():
    axes[1].text(row['Subtypes'], row['Silhouette'],
                 row['Fault_Type'][:15], fontsize=7, ha='center', va='bottom')
axes[1].set_xlabel('Number of Subtypes', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Subtypes vs Clustering Quality', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'phase09_clustering_quality.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Clustering quality plots saved")

#========================================
# 4. STATISTICAL SUMMARY
#========================================
print("\n" + "="*70)
print("4. STATISTICAL SUMMARY")
print("="*70)
print()

total_events = summary_df['Events'].sum()
total_subtypes = summary_df['Subtypes'].sum()
mean_silhouette = summary_df['Silhouette'].mean()
mean_subtypes_per_fault = summary_df['Subtypes'].mean()

print(f"Total Events Clustered: {total_events}")
print(f"Total Subtypes Discovered: {total_subtypes}")
print(f"Mean Subtypes per Fault: {mean_subtypes_per_fault:.1f}")
print(f"Mean Silhouette Score: {mean_silhouette:.3f}")
print()

print(f"Subtype Statistics:")
print(f"  Min subtypes: {summary_df['Subtypes'].min()}")
print(f"  Max subtypes: {summary_df['Subtypes'].max()}")
print(f"  Median subtypes: {summary_df['Subtypes'].median():.0f}")
print()

print(f"Silhouette Statistics:")
print(f"  Min: {summary_df['Silhouette'].min():.3f}")
print(f"  Max: {summary_df['Silhouette'].max():.3f}")
print(f"  Median: {summary_df['Silhouette'].median():.3f}")
print()

#========================================
# SUMMARY
#========================================
print("="*70)
print("SUMMARY AND CONCLUSIONS")
print("="*70)
print()

print(f"**Clustering Results**:")
print(f"  - {len(fault_types)} fault types analyzed")
print(f"  - {total_events} total events clustered")
print(f"  - {total_subtypes} distinct subtypes discovered")
print(f"  - Mean clustering quality (Silhouette): {mean_silhouette:.3f}")
print()

print(f"**Best Clustered Faults** (Silhouette > 0.50):")
best = summary_df[summary_df['Silhouette'] > 0.50].sort_values('Silhouette', ascending=False)
for _, row in best.iterrows():
    print(f"  - {row['Fault_Type']}: {row['Subtypes']} subtypes, Quality={row['Silhouette']:.3f}")
print()

print(f"**Interpretation**:")
print(f"  - Subtypes represent distinct fault mechanisms/severities")
print(f"  - High silhouette → clear mechanistic differences")
print(f"  - Low silhouette → continuous spectrum, not discrete types")
print(f"  - Examples: Hard quench vs soft quench, sudden vs gradual onset")
print()

print(f"**Practical Applications**:")
print(f"  1. Refined diagnostics: Identify specific failure modes")
print(f"  2. Severity classification: Prioritize by subtype")
print(f"  3. Targeted interventions: Subtype-specific fixes")
print(f"  4. Operator training: Recognize distinct fault signatures")
print()

print(f"**Next Steps**:")
print(f"  1. Manual inspection of cluster centers (prototypical events)")
print(f"  2. Physics validation: Do subtypes match known mechanisms?")
print(f"  3. Temporal analysis: Do subtypes have different onset patterns?")
print(f"  4. Operator feedback: Label subtypes with domain names")
print()

print("="*70)
print("✓ Phase 09 analysis complete!")
print(f"✓ All figures saved to: {FIGURES_DIR}")
print("="*70)
