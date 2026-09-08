#!/usr/bin/env python3
"""
Aggregate Batch Features for Analysis Notebooks
================================================

This script loads all batch files from prepare_data_cluster.py and creates
the aggregated files that the analysis notebooks expect:
- features_engineered.pkl: All features in DataFrame format
- preprocessed_data.pkl: Signals, derivatives, segments

The batch files already contain 1,243 engineered features per event!
This script just aggregates them into the expected format.

Usage:
    python aggregate_batch_features.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

print("="*80)
print("AGGREGATING BATCH FEATURES")
print("="*80)

# Paths
BATCHES_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data/batches')
OUTPUT_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')

# Find all batch files
batch_files = sorted(BATCHES_DIR.glob('batch_*.pkl'))
print(f"\nFound {len(batch_files)} batch files")

if len(batch_files) == 0:
    print("ERROR: No batch files found!")
    sys.exit(1)

# ============================================================================
# STEP 1: Load all batches and aggregate
# ============================================================================

print("\nStep 1: Loading batches...")
all_features = []
all_metadata = []
all_fault_labels_binary = []
all_fault_labels_multilabel = []

# We'll collect signals from first batch only to avoid memory issues
# (notebooks can load full sequences separately if needed)
collect_signals = True
signals_list = []
first_deriv_list = []
second_deriv_list = []

for batch_file in tqdm(batch_files, desc="Loading batches"):
    try:
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)

        results = batch_data.get('results', [])

        for event in results:
            if not event.get('success', False):
                continue

            # Features (already extracted!)
            features = event.get('features', {})
            all_features.append(features)

            # Metadata
            metadata = event.get('metadata', {})
            file_path = event.get('file', '')
            metadata['Fichier'] = file_path
            all_metadata.append(metadata)

            # Fault labels
            fault_info = event.get('fault_labels', {})
            all_fault_labels_binary.append(fault_info.get('binary', 0))
            all_fault_labels_multilabel.append(fault_info.get('multilabel', [0]*7))

            # Signals (only from first few batches to save memory)
            if collect_signals and len(signals_list) < 1000:
                signals = event.get('signals', {})
                if 'signals' in signals:
                    # Extract key signals as array
                    signal_names = ['Ucav', 'PhaseCav', 'Uci', 'PhaseUci',
                                   'A Ucr', 'A Uamp', 'vide', 'courant pickup']
                    signal_arrays = []
                    for sig_name in signal_names:
                        if sig_name in signals['signals']:
                            signal_arrays.append(signals['signals'][sig_name])

                    if signal_arrays:
                        signals_list.append(np.column_stack(signal_arrays))

                # Derivatives
                if 'first_deriv' in signals and 'Ucav' in signals['first_deriv']:
                    deriv_arrays = []
                    for sig_name in signal_names:
                        if sig_name in signals['first_deriv']:
                            deriv_arrays.append(signals['first_deriv'][sig_name])
                    if deriv_arrays:
                        first_deriv_list.append(np.column_stack(deriv_arrays))

                if 'second_deriv' in signals and 'Ucav' in signals['second_deriv']:
                    deriv_arrays = []
                    for sig_name in signal_names:
                        if sig_name in signals['second_deriv']:
                            deriv_arrays.append(signals['second_deriv'][sig_name])
                    if deriv_arrays:
                        second_deriv_list.append(np.column_stack(deriv_arrays))

    except Exception as e:
        print(f"  Warning: Error loading {batch_file.name}: {e}")
        continue

print(f"\n✓ Loaded data:")
print(f"  Events: {len(all_features)}")
print(f"  Features per event: {len(all_features[0]) if all_features else 0}")
print(f"  Signals collected: {len(signals_list)}")

# ============================================================================
# STEP 2: Create features DataFrame
# ============================================================================

print("\nStep 2: Creating features DataFrame...")
df_features = pd.DataFrame(all_features)

print(f"  Features DataFrame shape: {df_features.shape}")
print(f"  Features: {df_features.columns[:10].tolist()}...")

# ============================================================================
# STEP 3: Create metadata DataFrame
# ============================================================================

print("\nStep 3: Creating metadata DataFrame...")
df_metadata = pd.DataFrame(all_metadata)

# Add parsed timestamp
df_metadata['Timestamp'] = df_metadata['DATE'].apply(
    lambda x: pd.to_datetime(x, format="%b %d %Y %H:%M:%S", errors='coerce') if isinstance(x, str) else pd.NaT
)

# Parse cavity info from DBNAME
if 'DBNAME' in df_metadata.columns:
    df_metadata['Cryomodule'] = df_metadata['DBNAME'].str.split('-').str[1]
    df_metadata['Cavité'] = df_metadata['DBNAME'].str.split('-').str[2].str.split(':').str[0]
    df_metadata['ID'] = df_metadata['Cryomodule'] + '-' + df_metadata['Cavité']

print(f"  Metadata DataFrame shape: {df_metadata.shape}")

# ============================================================================
# STEP 4: Create fault label arrays
# ============================================================================

print("\nStep 4: Creating fault label arrays...")
y_binary = np.array(all_fault_labels_binary)
Y_multilabel = np.array(all_fault_labels_multilabel)

FAULT_COLUMNS = [
    'Seuil pick-up',
    'Coupure externe rapide',
    'Absence autorisation RF',
    'Seuil de vide',
    'Claquage ou quench cavité',
    'Dép seuil de sécurité RF',
    'Rég signal RF hors tolérance'
]

print(f"  Binary labels: {y_binary.shape}")
print(f"  Multi-label matrix: {Y_multilabel.shape}")
print(f"  Fault events: {y_binary.sum()}/{len(y_binary)} ({100*y_binary.sum()/len(y_binary):.1f}%)")

# ============================================================================
# STEP 5: Create signal arrays
# ============================================================================

print("\nStep 5: Creating signal arrays...")
if signals_list:
    signals_normalized = np.array(signals_list)
    print(f"  Signals shape: {signals_normalized.shape}")
else:
    signals_normalized = None
    print(f"  No signals collected (will need to load from batches separately)")

if first_deriv_list:
    first_deriv_normalized = np.array(first_deriv_list)
    print(f"  First derivatives shape: {first_deriv_normalized.shape}")
else:
    first_deriv_normalized = None

if second_deriv_list:
    second_deriv_normalized = np.array(second_deriv_list)
    print(f"  Second derivatives shape: {second_deriv_normalized.shape}")
else:
    second_deriv_normalized = None

# ============================================================================
# STEP 6: Save features_engineered.pkl (for ML notebooks)
# ============================================================================

print("\nStep 6: Saving features_engineered.pkl...")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale features
scaler = StandardScaler()
feature_cols = df_features.columns.tolist()
X_scaled = scaler.fit_transform(df_features)

# PCA (keep 95% variance)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"  PCA: {X_scaled.shape[1]} features → {X_pca.shape[1]} components")
print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

feature_data = {
    # Features
    'features_all': df_features,
    'feature_cols': feature_cols,
    'X_scaled': X_scaled,
    'X_pca': X_pca,

    # Sequences (limited to first 1000 for memory)
    'sequences_full': signals_normalized,
    'sequences_pretrigger': signals_normalized[:, :3000, :] if signals_normalized is not None else None,

    # Targets
    'y_binary': y_binary,
    'y_multilabel': Y_multilabel,
    'fault_column_names': FAULT_COLUMNS,

    # Transformers
    'scaler': scaler,
    'pca': pca,

    # Metadata
    'metadata': df_metadata,
    'signal_names': ['Ucav', 'PhaseCav', 'Uci', 'PhaseUci',
                     'A Ucr', 'A Uamp', 'vide', 'courant pickup'],
}

features_file = OUTPUT_DIR / 'features_engineered.pkl'
with open(features_file, 'wb') as f:
    pickle.dump(feature_data, f, protocol=4)

file_size_mb = features_file.stat().st_size / (1024**2)
print(f"  ✓ Saved: {features_file}")
print(f"  Size: {file_size_mb:.2f} MB")

# ============================================================================
# STEP 7: Save preprocessed_data.pkl (for preprocessing notebooks)
# ============================================================================

print("\nStep 7: Saving preprocessed_data.pkl...")

CONFIG = {
    'delta_pre': 3000,
    'delta_post': 1000,
}

preprocessed_data = {
    'signals_normalized': signals_normalized,
    'first_derivative_normalized': first_deriv_normalized,
    'second_derivative_normalized': second_deriv_normalized,
    'segmented_signals': {},  # Empty for now (can be loaded from batches)
    'metadata': df_metadata.to_dict('records'),
    'signal_names': ['Ucav', 'PhaseCav', 'Uci', 'PhaseUci',
                     'A Ucr', 'A Uamp', 'vide', 'courant pickup'],
    'config': CONFIG,
}

preprocessed_file = OUTPUT_DIR / 'preprocessed_data.pkl'
with open(preprocessed_file, 'wb') as f:
    pickle.dump(preprocessed_data, f, protocol=4)

file_size_mb = preprocessed_file.stat().st_size / (1024**2)
print(f"  ✓ Saved: {preprocessed_file}")
print(f"  Size: {file_size_mb:.2f} MB")

# ============================================================================
# STEP 8: Save fault labels
# ============================================================================

print("\nStep 8: Saving fault labels...")

fault_matrix_file = OUTPUT_DIR / 'fault_labels_matrix.npy'
np.save(fault_matrix_file, Y_multilabel)
print(f"  ✓ Saved: {fault_matrix_file}")

fault_names_file = OUTPUT_DIR / 'fault_column_names.pkl'
with open(fault_names_file, 'wb') as f:
    pickle.dump(FAULT_COLUMNS, f)
print(f"  ✓ Saved: {fault_names_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("AGGREGATION COMPLETE!")
print("="*80)

print(f"\nDataset Summary:")
print(f"  Total events: {len(all_features)}")
print(f"  Features extracted: {df_features.shape[1]}")
print(f"  PCA components: {X_pca.shape[1]}")
print(f"  Fault events: {y_binary.sum()} ({100*y_binary.sum()/len(y_binary):.1f}%)")
print(f"  Normal events: {(y_binary==0).sum()} ({100*(y_binary==0).sum()/len(y_binary):.1f}%)")

print(f"\nFault distribution:")
for i, fault_name in enumerate(FAULT_COLUMNS):
    count = Y_multilabel[:, i].sum()
    pct = 100 * count / len(Y_multilabel)
    print(f"  {fault_name:35s}: {count:5d} events ({pct:5.1f}%)")

print(f"\nFiles created:")
print(f"  ✓ {features_file}")
print(f"  ✓ {preprocessed_file}")
print(f"  ✓ {fault_matrix_file}")
print(f"  ✓ {fault_names_file}")

print(f"\n✅ Analysis notebooks can now load the aggregated data!")
print("="*80)
