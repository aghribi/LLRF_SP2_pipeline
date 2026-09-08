#!/usr/bin/env python3
"""
Notebook 01: Data Overview Analysis
Analysis-only version that loads processed batch data.

This script analyzes metadata and fault labels from the cluster pipeline
WITHOUT reprocessing raw binary files.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

print("="*70)
print("NOTEBOOK 01: DATA OVERVIEW ANALYSIS")
print("="*70)

# ========================================
# Setup
# ========================================
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

OUTPUT_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')
BATCHES_DIR = OUTPUT_DIR / 'batches'

# ========================================
# 1. Load Data from Batches
# ========================================
print("\n## 1. Loading Processed Data from Batches...")
print("="*70)

# Find all batch files
batch_files = sorted(BATCHES_DIR.glob('batch_*.pkl'))
print(f"Found {len(batch_files)} batch files")

if len(batch_files) == 0:
    print("❌ No batch files found. Run the cluster pipeline first:")
    print("   ./run_pipeline.sh")
    exit(1)

# Load batch metadata to count total events
print("\nScanning batches...")
batch_info = []
total_events = 0

for batch_file in batch_files[:5]:  # Load first 5 batches for initial analysis
    try:
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)

        # Batch is a dict with 'results', 'n_events', 'batch_idx'
        if isinstance(batch_data, dict):
            n_events = batch_data.get('n_events', 0)
        else:
            n_events = len(batch_data) if isinstance(batch_data, list) else 0

        total_events += n_events

        batch_info.append({
            'batch': batch_file.stem,
            'events': n_events,
            'size_mb': batch_file.stat().st_size / (1024**2)
        })

        print(f"  {batch_file.stem}: {n_events} events ({batch_file.stat().st_size / (1024**2):.1f} MB)")
    except Exception as e:
        print(f"  ⚠️  Error loading {batch_file.stem}: {e}")

print(f"\nTotal events in first 5 batches: {total_events}")

# ========================================
# 2. Aggregate Metadata from First Batch (for quick analysis)
# ========================================
print("\n## 2. Aggregating Metadata...")
print("="*70)

# Load first batch for detailed analysis
with open(batch_files[0], 'rb') as f:
    batch_0 = pickle.load(f)

print(f"\nFirst batch structure:")
print(f"  Type: {type(batch_0)}")

# Extract events from batch dict
if isinstance(batch_0, dict):
    events = batch_0.get('results', [])
    print(f"  Events: {len(events)}")
else:
    events = batch_0
    print(f"  Events: {len(events)}")

if events:
    print(f"  Sample event keys: {list(events[0].keys())[:15]}")

# Create DataFrame from first batch
# Each event has nested 'metadata' dict
metadata_list = []

for event in events[:500]:  # Analyze first 500 events
    if event.get('success') and 'metadata' in event:
        # Extract the metadata dict
        meta = event['metadata'].copy()
        # Add file path
        meta['Fichier'] = event.get('file', '')
        metadata_list.append(meta)

df_metadata = pd.DataFrame(metadata_list)

print(f"\n✓ Created metadata DataFrame:")
print(f"  Shape: {df_metadata.shape}")
print(f"  Columns: {list(df_metadata.columns[:15])}...")

# ========================================
# 3. Parse Fault Labels from ALM Field
# ========================================
print("\n## 3. Fault Label Analysis...")
print("="*70)

# Expected fault column names (from original notebook)
FAULT_COLUMNS = [
    'Seuil pick-up',
    'Coupure externe rapide',
    'Absence autorisation RF',
    'Seuil de vide',
    'Claquage ou quench cavité',
    'Dép seuil de sécurité RF',
    'Rég signal RF hors tolérance'
]

# Parse ALM field to extract multi-label fault flags
print("Parsing ALM field to extract fault labels...")

fault_labels_list = []

for idx, row in df_metadata.iterrows():
    alm_hex = row.get('ALM', '0x0000')

    try:
        # Convert hex ALM to integer
        if isinstance(alm_hex, str):
            alm_value = int(alm_hex, 16)
        else:
            alm_value = int(alm_hex)

        # Extract 7 fault bits (LSB first)
        alm_binary = format(alm_value, '07b')[::-1]
        fault_flags = [int(b) for b in alm_binary]

        # Ensure we have exactly 7 flags
        while len(fault_flags) < 7:
            fault_flags.append(0)

        fault_labels_list.append(fault_flags[:7])

    except Exception as e:
        # Default to no faults if parsing fails
        fault_labels_list.append([0] * 7)

# Create fault label matrix
Y_multilabel = np.array(fault_labels_list)

# Add fault columns to metadata
for i, fault_name in enumerate(FAULT_COLUMNS):
    df_metadata[fault_name] = Y_multilabel[:, i]

print(f"\n✓ Fault label matrix created:")
print(f"  Shape: {Y_multilabel.shape}")
print(f"  Fault events: {(Y_multilabel.sum(axis=1) > 0).sum()}")
print(f"  Normal events: {(Y_multilabel.sum(axis=1) == 0).sum()}")

# Fault distribution
print(f"\nFault type distribution:")
for i, fault_name in enumerate(FAULT_COLUMNS):
    count = Y_multilabel[:, i].sum()
    pct = 100 * count / len(Y_multilabel) if len(Y_multilabel) > 0 else 0
    print(f"  {fault_name:35s}: {count:4d} events ({pct:5.1f}%)")

# Co-occurrence analysis
df_cooccur = pd.DataFrame(
    Y_multilabel.T @ Y_multilabel,
    index=FAULT_COLUMNS,
    columns=FAULT_COLUMNS
)

print(f"\n✓ Fault co-occurrence matrix created: {df_cooccur.shape}")

# ========================================
# 4. Metadata Statistics
# ========================================
print("\n## 4. Metadata Statistics...")
print("="*70)

print(f"\nDataset information:")
print(f"  Total events (first batch sample): {len(df_metadata)}")
print(f"  Columns: {df_metadata.shape[1]}")
print(f"  Memory usage: {df_metadata.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Show basic statistics
print(f"\nKey parameters:")
if 'NDEC' in df_metadata.columns:
    print(f"  NDEC values: {df_metadata['NDEC'].value_counts().to_dict()}")

if 'LOOP' in df_metadata.columns:
    print(f"  LOOP states: {df_metadata['LOOP'].value_counts().to_dict()}")

if 'KPI' in df_metadata.columns:
    # Convert KPI to numeric (may be stored as string)
    df_metadata['KPI'] = pd.to_numeric(df_metadata['KPI'], errors='coerce')
    print(f"  KPI (proportional gain):")
    print(f"    Mean: {df_metadata['KPI'].mean():.2f}")
    print(f"    Range: [{df_metadata['KPI'].min():.2f}, {df_metadata['KPI'].max():.2f}]")

# ========================================
# 5. Temporal Distribution
# ========================================
print("\n## 5. Temporal Distribution...")
print("="*70)

if 'Timestamp' in df_metadata.columns:
    # Convert timestamp
    df_metadata['Timestamp'] = pd.to_datetime(df_metadata['Timestamp'], errors='coerce')

    valid_timestamps = df_metadata['Timestamp'].notna().sum()
    print(f"  Valid timestamps: {valid_timestamps}/{len(df_metadata)}")

    if valid_timestamps > 0:
        print(f"  Date range: {df_metadata['Timestamp'].min()} to {df_metadata['Timestamp'].max()}")

        # Events by year
        df_metadata['Year'] = df_metadata['Timestamp'].dt.year
        print(f"\nEvents by year:")
        for year, count in sorted(df_metadata['Year'].value_counts().items()):
            print(f"    {year}: {count} events")

# ========================================
# 6. Cavity Distribution
# ========================================
print("\n## 6. Cavity Distribution...")
print("="*70)

# Parse cavity info from DBNAME
if 'DBNAME' in df_metadata.columns:
    print("\nParsing cavity information from DBNAME...")

    # Extract cryomodule and cavity
    df_metadata['Cryomodule'] = df_metadata['DBNAME'].str.split('-').str[1]
    df_metadata['Cavité'] = df_metadata['DBNAME'].str.split('-').str[2].str.split(':').str[0]
    df_metadata['ID'] = df_metadata['Cryomodule'] + '-' + df_metadata['Cavité']

    print(f"  Unique cavities: {df_metadata['ID'].nunique()}")
    print(f"\nTop 10 cavities by event count:")
    for cavity, count in df_metadata['ID'].value_counts().head(10).items():
        print(f"    {cavity:15s}: {count:4d} events")

# ========================================
# 7. Save Aggregated Data
# ========================================
print("\n## 7. Saving Analysis Results...")
print("="*70)

# Save metadata sample
metadata_file = OUTPUT_DIR / f'metadata_sample_{len(df_metadata)}_events.csv'
df_metadata.to_csv(metadata_file, index=False)
print(f"✓ Saved metadata: {metadata_file}")

# Save fault labels
if Y_multilabel is not None:
    np.save(OUTPUT_DIR / 'fault_labels_matrix.npy', Y_multilabel)
    print(f"✓ Saved fault labels: fault_labels_matrix.npy")

    with open(OUTPUT_DIR / 'fault_column_names.pkl', 'wb') as f:
        pickle.dump(FAULT_COLUMNS, f)
    print(f"✓ Saved fault column names")

# ========================================
# SUMMARY
# ========================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ Loaded and analyzed {len(batch_files)} batch files")
print(f"✓ Processed {len(df_metadata)} events from first batch")
print(f"✓ Fault types: {len(FAULT_COLUMNS)}")
if Y_multilabel is not None:
    print(f"✓ Fault events: {(Y_multilabel.sum(axis=1) > 0).sum()}")
    print(f"✓ Normal events: {(Y_multilabel.sum(axis=1) == 0).sum()}")
print("="*70)

print("\n✅ Data overview analysis complete!")
print("\nNext steps:")
print("  1. The full pipeline is still running (check llrf_prep_19118402.out)")
print("  2. Once complete, re-run this script to analyze all batches")
print("  3. Proceed to notebook 02 (Signal Visualization)")
