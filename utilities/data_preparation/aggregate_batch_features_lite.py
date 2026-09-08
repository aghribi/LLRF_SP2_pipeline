#!/usr/bin/env python3
"""
Aggregate Batch Features - Memory-Efficient Version
====================================================

Loads only features and metadata from batches, skipping the huge signal arrays.
Analysis notebooks can load signals separately from batches if needed.

Usage:
    python aggregate_batch_features_lite.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import gc

print("="*80)
print("AGGREGATING BATCH FEATURES (LITE VERSION)")
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
# STEP 1: Load features and metadata only (skip signals)
# ============================================================================

print("\nStep 1: Loading features and metadata (skipping signals)...")
all_features = []
all_metadata = []
all_fault_labels_binary = []
all_fault_labels_multilabel = []

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

        # Free memory
        del batch_data
        del results
        gc.collect()

    except Exception as e:
        print(f"\n  Warning: Error loading {batch_file.name}: {e}")
        continue

print(f"\n✓ Loaded data:")
print(f"  Events: {len(all_features)}")
print(f"  Features per event: {len(all_features[0]) if all_features else 0}")

# ============================================================================
# STEP 2: Create features DataFrame
# ============================================================================

print("\nStep 2: Creating features DataFrame...")
df_features = pd.DataFrame(all_features)

print(f"  Features DataFrame shape: {df_features.shape}")
print(f"  Sample features: {df_features.columns[:10].tolist()}...")

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
print(f"  Date range: {df_metadata['Timestamp'].min()} to {df_metadata['Timestamp'].max()}")

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
print(f"  Normal events: {(y_binary==0).sum()} ({100*(y_binary==0).sum()/len(y_binary):.1f}%)")

# ============================================================================
# STEP 5: Save features_engineered.pkl (for ML notebooks)
# ============================================================================

print("\nStep 5: Preparing feature data...")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale features
print("  Scaling features...")
scaler = StandardScaler()
feature_cols = df_features.columns.tolist()
X_scaled = scaler.fit_transform(df_features)

# PCA (keep 95% variance)
print("  Running PCA...")
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

    # NOTE: Sequences not included to save memory
    # Load from batch files separately if needed
    'sequences_full': None,
    'sequences_pretrigger': None,

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

print("\nStep 6: Saving features_engineered.pkl...")
features_file = OUTPUT_DIR / 'features_engineered.pkl'
with open(features_file, 'wb') as f:
    pickle.dump(feature_data, f, protocol=4)

file_size_mb = features_file.stat().st_size / (1024**2)
print(f"  ✓ Saved: {features_file}")
print(f"  Size: {file_size_mb:.2f} MB")

# ============================================================================
# STEP 7: Save fault labels
# ============================================================================

print("\nStep 7: Saving fault labels...")

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

print(f"\nTemporal coverage:")
years = df_metadata['Timestamp'].dt.year.value_counts().sort_index()
for year, count in years.items():
    print(f"  {year}: {count} events")

print(f"\nCavities:")
cavity_counts = df_metadata['ID'].value_counts().head(10)
for cavity, count in cavity_counts.items():
    print(f"  {cavity}: {count} events")

print(f"\nFiles created:")
print(f"  ✓ {features_file}")
print(f"  ✓ {fault_matrix_file}")
print(f"  ✓ {fault_names_file}")

print(f"\n📝 NOTE: Signal arrays not saved (too large, ~400GB)")
print(f"   Analysis notebooks will load signals from batch files as needed")

print(f"\n✅ Analysis notebooks can now load the aggregated features!")
print("="*80)
