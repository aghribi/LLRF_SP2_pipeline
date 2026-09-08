#!/usr/bin/env python
"""
Execute notebook 01 cells manually to avoid jupyter execution issues
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
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("NOTEBOOK 01: DATA OVERVIEW ANALYSIS")
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
signal_names = data['signal_names']
sequences_full = data.get('sequences_full', None)

# Convert metadata to DataFrame if it's a list
if isinstance(metadata, list):
    df_metadata = pd.DataFrame(metadata)
else:
    df_metadata = metadata

print(f"\n✓ Dataset loaded successfully")
print(f"  Total events: {len(df_metadata):,}")
print(f"  Features: {len(feature_cols):,}")
print(f"  PCA components: {X_pca.shape[1]}")
print(f"  Signals: {len(signal_names)}")
print(f"  Fault types: {len(fault_names)}")
print(f"  Sequences shape: {sequences_full.shape if sequences_full is not None else 'N/A'}")

# Dataset Statistics
n_fault = y_binary.sum()
n_normal = (y_binary == 0).sum()
total = len(y_binary)

print("\n" + "="*70)
print("DATASET OVERVIEW")
print("="*70)
print(f"\nTotal Events: {total:,}")
print(f"  Normal events (ALM=0): {n_normal:,} ({100*n_normal/total:.1f}%)")
print(f"  Fault events (ALM>0):  {n_fault:,} ({100*n_fault/total:.1f}%)")
print(f"\nClass Ratio: {n_normal/n_fault:.2f}:1 (Normal:Fault)")

# Fault distribution
print(f"\n" + "="*70)
print("FAULT TYPE DISTRIBUTION")
print("="*70)
for i, fault_name in enumerate(fault_names):
    count = Y_multilabel[:, i].sum()
    pct = 100 * count / total
    print(f"  {fault_name:40s}: {count:4d} events ({pct:5.1f}%)")

# Multi-label statistics
n_faults_per_event = Y_multilabel.sum(axis=1)
single_fault = (n_faults_per_event == 1).sum()
multi_fault = (n_faults_per_event > 1).sum()

print(f"\n" + "="*70)
print("MULTI-LABEL CHARACTERISTICS")
print("="*70)
print(f"  Single fault events: {single_fault:,} ({100*single_fault/n_fault:.1f}% of faults)")
print(f"  Multi-fault events:  {multi_fault:,} ({100*multi_fault/n_fault:.1f}% of faults)")
print(f"  Average faults per fault event: {n_faults_per_event[y_binary==1].mean():.2f}")
print(f"  Max concurrent faults: {int(n_faults_per_event.max())}")

# Visualizations
print(f"\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# 1. Fault distribution bar chart
fault_counts = pd.Series({name: Y_multilabel[:, i].sum() for i, name in enumerate(fault_names)})
fault_counts = fault_counts.sort_values(ascending=False)

plt.figure(figsize=(14, 6))
fault_counts.plot(kind='barh', color='coral', edgecolor='black')
plt.xlabel('Number of Events', fontsize=12)
plt.title(f'Fault Type Distribution (N={total:,} events)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
output_file = OUTPUT_DIR / '01_fault_distribution.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_file}")

# 2. Fault co-occurrence
df_cooccur = pd.DataFrame(
    Y_multilabel.T @ Y_multilabel,
    index=fault_names,
    columns=fault_names
)

# Conditional probability P(j|i)
fault_totals = Y_multilabel.sum(axis=0)
df_cond_prob = df_cooccur.copy()
for i in range(len(fault_names)):
    if fault_totals[i] > 0:
        df_cond_prob.iloc[i, :] = df_cooccur.iloc[i, :] / fault_totals[i]
    else:
        df_cond_prob.iloc[i, :] = 0

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(df_cooccur, annot=True, fmt='g', cmap='YlOrRd', ax=axes[0],
            cbar_kws={'label': 'Count'}, square=True)
axes[0].set_title('Fault Co-occurrence Matrix', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Fault Type')
axes[0].set_ylabel('Fault Type')

sns.heatmap(df_cond_prob, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
            cbar_kws={'label': 'P(j|i)'}, vmin=0, vmax=1, square=True)
axes[1].set_title('Conditional Probability P(column|row)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Fault Type (j)')
axes[1].set_ylabel('Given Fault Type (i)')

plt.tight_layout()
output_file = OUTPUT_DIR / '01_fault_cooccurrence.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_file}")

# 3. Temporal analysis (if available)
if 'Timestamp' in df_metadata.columns or 'DATE' in df_metadata.columns:
    print("\nProcessing temporal data...")

    if 'Timestamp' not in df_metadata.columns and 'DATE' in df_metadata.columns:
        df_metadata['Timestamp'] = df_metadata['DATE'].apply(
            lambda x: pd.to_datetime(x, format="%b %d %Y %H:%M:%S", errors='coerce') if isinstance(x, str) else pd.NaT
        )
    else:
        df_metadata['Timestamp'] = pd.to_datetime(df_metadata['Timestamp'], errors='coerce')

    valid_timestamps = df_metadata['Timestamp'].notna().sum()

    if valid_timestamps > 100:
        print(f"Valid timestamps: {valid_timestamps}/{len(df_metadata)}")
        print(f"Date range: {df_metadata['Timestamp'].min()} to {df_metadata['Timestamp'].max()}")

        df_metadata['Year'] = df_metadata['Timestamp'].dt.year
        df_metadata['Month'] = df_metadata['Timestamp'].dt.to_period('M')

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Events by year
        year_counts = df_metadata['Year'].value_counts().sort_index()
        year_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Number of Events', fontsize=12)
        axes[0].set_title('Event Distribution by Year', fontweight='bold', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='y')

        # Timeline plot
        month_counts = df_metadata.groupby('Month').size().sort_index()
        month_counts.plot(ax=axes[1], color='coral', marker='o', linewidth=2)
        axes[1].set_xlabel('Month', fontsize=12)
        axes[1].set_ylabel('Number of Events', fontsize=12)
        axes[1].set_title('Event Timeline', fontweight='bold', fontsize=14)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = OUTPUT_DIR / '01_temporal_distribution.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_file}")
else:
    print("\nNo temporal data available")

# Cavity analysis
if 'DBNAME' in df_metadata.columns:
    print("\nProcessing cavity information...")
    df_metadata['Cryomodule'] = df_metadata['DBNAME'].str.split('-').str[1]
    df_metadata['Cavity'] = df_metadata['DBNAME'].str.split('-').str[2].str.split(':').str[0]
    df_metadata['Cavity_ID'] = df_metadata['Cryomodule'] + '-' + df_metadata['Cavity']

    n_cavities = df_metadata['Cavity_ID'].nunique()
    n_cryomodules = df_metadata['Cryomodule'].nunique()

    print(f"Cavity Statistics:")
    print(f"  Unique cavities: {n_cavities}")
    print(f"  Unique cryomodules: {n_cryomodules}")
    print(f"\nTop 10 cavities by event count:")
    for cavity, count in df_metadata['Cavity_ID'].value_counts().head(10).items():
        pct = 100 * count / len(df_metadata)
        print(f"  {cavity:15s}: {count:4d} events ({pct:5.1f}%)")

# Save summary report
print(f"\n" + "="*70)
print("SAVING SUMMARY REPORT")
print("="*70)

summary_file = OUTPUT_DIR / '01_data_loading_summary.txt'

with open(summary_file, 'w') as f:
    f.write("LLRF Anomaly Detection Dataset Summary\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("Dataset Overview:\n")
    f.write(f"  Total events: {total:,}\n")
    f.write(f"  Normal events: {n_normal:,} ({100*n_normal/total:.1f}%)\n")
    f.write(f"  Fault events: {n_fault:,} ({100*n_fault/total:.1f}%)\n")
    f.write(f"  Features extracted: {len(feature_cols):,}\n")
    f.write(f"  PCA components: {X_pca.shape[1]}\n")
    f.write(f"  Signals: {len(signal_names)}\n\n")

    f.write("Fault Type Distribution:\n")
    for i, fault_name in enumerate(fault_names):
        count = Y_multilabel[:, i].sum()
        pct = 100 * count / total
        f.write(f"  {fault_name:40s}: {count:4d} events ({pct:5.1f}%)\n")

    f.write("\nMulti-label Statistics:\n")
    f.write(f"  Single fault events: {single_fault:,}\n")
    f.write(f"  Multi-fault events: {multi_fault:,}\n")
    f.write(f"  Avg faults per fault event: {n_faults_per_event[y_binary==1].mean():.2f}\n")
    f.write(f"  Max concurrent faults: {int(n_faults_per_event.max())}\n")

    if 'Cavity_ID' in df_metadata.columns:
        f.write("\nCavity Coverage:\n")
        f.write(f"  Unique cavities: {df_metadata['Cavity_ID'].nunique()}\n")
        f.write(f"  Unique cryomodules: {df_metadata['Cryomodule'].nunique()}\n")

print(f"✓ Saved summary: {summary_file}")

print("\n" + "="*70)
print("NOTEBOOK 01 COMPLETE")
print("="*70)
print(f"\nKey Results:")
print(f"  - Analyzed {total:,} events")
print(f"  - {n_fault:,} fault events, {n_normal:,} normal events")
print(f"  - {len(feature_cols):,} features across {len(signal_names)} signals")
print(f"  - Generated visualizations saved to {OUTPUT_DIR}")
print("\n✓ All outputs generated successfully")
