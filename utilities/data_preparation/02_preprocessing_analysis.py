#!/usr/bin/env python3
"""
Preprocessing Analysis - Load signals from batch data and apply preprocessing
This version works with pre-processed batch data from the cluster pipeline.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.signal import butter, filtfilt

print("="*70)
print("Preprocessing Analysis - Batch Data Version")
print("="*70)

# ========================================
# 1. Setup
# ========================================
OUTPUT_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')
BATCHES_DIR = OUTPUT_DIR / 'batches'

# Configuration
CONFIG = {
    'cutoff_freq': 10,      # High-pass filter cutoff (Hz)
    'filter_order': 1,      # Butterworth filter order
}

print("\n## 1. Loading Batch Data...")
print("="*70)

# Find batch files
batch_files = sorted(BATCHES_DIR.glob('batch_*.pkl'))
print(f"Found {len(batch_files)} batch files")

if len(batch_files) == 0:
    print("❌ No batch files found. Run pipeline first.")
    exit(1)

# ========================================
# 2. Load signals from first batch
# ========================================
print("\n## 2. Extracting Signals from Batches...")
print("="*70)

# Load first batch for analysis
with open(batch_files[0], 'rb') as f:
    batch_0 = pickle.load(f)

events = batch_0.get('results', [])
print(f"First batch: {len(events)} events")

# Extract signals from first few events
signals_list = []
metadata_list = []

for i, event in enumerate(events[:50]):  # Analyze first 50 events
    if not event.get('success'):
        continue

    signals_dict = event.get('signals', {})
    if not signals_dict:
        continue

    # Get signal names
    if i == 0:
        signal_names = list(signals_dict.keys())
        print(f"\nAvailable signals: {len(signal_names)}")
        for j, name in enumerate(signal_names[:10], 1):
            print(f"  {j}. {name}")
        if len(signal_names) > 10:
            print(f"  ... and {len(signal_names) - 10} more")

    # Stack signals into array
    try:
        signal_arrays = [signals_dict[name] for name in signal_names]
        # Find minimum length
        min_len = min(len(arr) for arr in signal_arrays)
        # Truncate all to same length
        signal_arrays = [arr[:min_len] for arr in signal_arrays]
        signal_matrix = np.column_stack(signal_arrays)

        signals_list.append(signal_matrix)
        metadata_list.append(event.get('metadata', {}))
    except Exception as e:
        print(f"  ⚠️  Error processing event {i}: {e}")
        continue

print(f"\n✓ Extracted {len(signals_list)} events")

if len(signals_list) == 0:
    print("❌ No signals extracted")
    exit(1)

# ========================================
# 3. Stack into 3D array
# ========================================
print("\n## 3. Stacking Signals...")
print("="*70)

# Check shapes
shapes = [s.shape for s in signals_list]
print(f"Signal shapes: {set(shapes)}")

# Find most common shape
from collections import Counter
shape_counts = Counter(shapes)
target_shape = shape_counts.most_common(1)[0][0]
print(f"Target shape: {target_shape}")

# Filter to events with target shape
filtered_signals = []
filtered_metadata = []

for sig, meta in zip(signals_list, metadata_list):
    if sig.shape == target_shape:
        filtered_signals.append(sig)
        filtered_metadata.append(meta)

signals_array = np.array(filtered_signals)
print(f"\n✓ Stacked array shape: {signals_array.shape}")
print(f"  Format: (n_events, n_samples, n_signals)")

# ========================================
# 4. Normalization
# ========================================
print("\n## 4. Z-Score Normalization...")
print("="*70)

# Compute global mean and std
signal_means = signals_array.mean(axis=(0, 1))
signal_stds = signals_array.std(axis=(0, 1))

print("\nNormalization statistics:")
print(f"{'Signal':<30s} {'Mean':>15s} {'Std':>15s}")
print("-" * 65)
for i, name in enumerate(signal_names[:len(signal_means)]):
    print(f"{name:<30s} {signal_means[i]:>15.4e} {signal_stds[i]:>15.4e}")

# Apply normalization
signals_normalized = (signals_array - signal_means) / (signal_stds + 1e-10)

print(f"\n✓ Normalization applied")
print(f"  Normalized shape: {signals_normalized.shape}")

# Verify
norm_means = signals_normalized.mean(axis=(0, 1))
norm_stds = signals_normalized.std(axis=(0, 1))

max_mean_dev = np.abs(norm_means).max()
print(f"  Max mean deviation from 0: {max_mean_dev:.2e}")
print(f"  Mean std deviation from 1: {np.abs(norm_stds - 1).mean():.3f}")

# ========================================
# 5. Save preprocessed data
# ========================================
print("\n## 5. Saving Preprocessed Data...")
print("="*70)

preprocessed_data = {
    'signals_raw': signals_array,
    'signals_normalized': signals_normalized,
    'signal_names': signal_names,
    'metadata': filtered_metadata,
    'normalization': {
        'means': signal_means,
        'stds': signal_stds,
    },
    'config': CONFIG,
    'shape': signals_array.shape,
}

output_file = OUTPUT_DIR / 'preprocessed_signals.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(preprocessed_data, f, protocol=4)

file_size_mb = output_file.stat().st_size / (1024**2)

print(f"✓ Saved: {output_file}")
print(f"  Size: {file_size_mb:.2f} MB")
print(f"  Events: {len(filtered_signals)}")
print(f"  Signals: {len(signal_names)}")

print("\n" + "="*70)
print("PREPROCESSING COMPLETE")
print("="*70)
print("✓ Signals extracted from batch data")
print("✓ Normalization applied")
print("✓ Data ready for feature engineering")
print("="*70)
