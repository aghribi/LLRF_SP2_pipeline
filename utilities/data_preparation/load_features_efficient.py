#!/usr/bin/env python3
"""
Memory-Efficient Feature Loader - Loads ALL batches without OOM
================================================================

Extracts only features, metadata, and labels from batches (NOT raw signals).
This reduces memory usage from ~300GB to <100MB!

Usage:
    python load_features_efficient.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc

print("="*80)
print("MEMORY-EFFICIENT FEATURE LOADER - All Batches")
print("="*80)

# Paths
BATCHES_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data/batches')
OUTPUT_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')

# Find all batch files
batch_files = sorted(BATCHES_DIR.glob('batch_*.pkl'))
print(f"\nFound {len(batch_files)} batch files")
print(f"Processing incrementally to avoid OOM...")

# Accumulators (store ONLY features, not signals)
all_features = []
all_metadata = []
all_fault_labels_binary = []
all_fault_labels_multilabel = []

# Process batches one at a time
for i, batch_file in enumerate(tqdm(batch_files, desc="Loading batches")):
    # Load batch
    with open(batch_file, 'rb') as f:
        batch_data = pickle.load(f)

    results = batch_data.get('results', [])

    # Extract only what we need (NOT raw signals!)
    for event in results:
        if not event.get('success', False):
            continue

        # Extract features (small dict of floats)
        all_features.append(event.get('features', {}))

        # Extract metadata
        metadata = event.get('metadata', {}).copy()
        metadata['Fichier'] = event.get('file', '')
        all_metadata.append(metadata)

        # Extract fault labels
        fault_info = event.get('fault_labels', {})
        all_fault_labels_binary.append(fault_info.get('binary', 0))
        all_fault_labels_multilabel.append(fault_info.get('multilabel', [0]*7))

    # Clear batch from memory immediately
    del batch_data, results
    gc.collect()

    # Progress update every 5 batches
    if (i + 1) % 5 == 0:
        print(f"  [{i+1}/{len(batch_files)}] Loaded {len(all_features)} events so far...")

print(f"\n✓ Loaded {len(all_features)} events total")
print(f"  Features per event: {len(all_features[0])}")

# Create DataFrames
print("\nCreating DataFrames...")
df_features = pd.DataFrame(all_features)
df_metadata = pd.DataFrame(all_metadata)

# Parse timestamps
print("Parsing timestamps...")
df_metadata['Timestamp'] = df_metadata['DATE'].apply(
    lambda x: pd.to_datetime(x, format="%b %d %Y %H:%M:%S", errors='coerce') if isinstance(x, str) else pd.NaT
)

# Parse cavity info
if 'DBNAME' in df_metadata.columns:
    df_metadata['Cryomodule'] = df_metadata['DBNAME'].str.split('-').str[1]
    df_metadata['Cavité'] = df_metadata['DBNAME'].str.split('-').str[2].str.split(':').str[0]
    df_metadata['ID'] = df_metadata['Cryomodule'] + '-' + df_metadata['Cavité']

# Create labels
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

# Handle missing values
print(f"\nHandling missing values...")
n_missing = df_features.isnull().sum().sum()
print(f"  Found {n_missing} missing values")

if n_missing > 0:
    # Fill NaN with median
    df_features = df_features.fillna(df_features.median())
    # If still NaN (all-NaN columns), fill with 0
    df_features = df_features.fillna(0)
    print(f"  ✓ Filled missing values with median/0")

# Scale and PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print(f"\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

print(f"Running PCA...")
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"  PCA: {X_scaled.shape[1]} features → {X_pca.shape[1]} components ({pca.explained_variance_ratio_.sum():.1%} variance)")

# Save
print(f"\nSaving to disk...")
feature_data = {
    'features_all': df_features,
    'feature_cols': df_features.columns.tolist(),
    'X_scaled': X_scaled,
    'X_pca': X_pca,
    'y_binary': y_binary,
    'y_multilabel': Y_multilabel,
    'fault_column_names': FAULT_COLUMNS,
    'scaler': scaler,
    'pca': pca,
    'metadata': df_metadata,
    'signal_names': ['Ucav', 'PhaseCav', 'Uci', 'PhaseUci', 'A Ucr', 'A Uamp', 'vide', 'courant pickup'],
    'sequences_full': None,  # Not loading raw signals to save memory
    'sequences_pretrigger': None,
}

output_file = OUTPUT_DIR / 'features_engineered.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(feature_data, f, protocol=4)

print(f"\n✓ SUCCESS! Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / (1024**2):.1f} MB")
print(f"  Events: {len(y_binary)}")
print(f"  Features: {X_scaled.shape[1]}")
print(f"  PCA components: {X_pca.shape[1]}")

# Save fault labels
np.save(OUTPUT_DIR / 'fault_labels_matrix.npy', Y_multilabel)

# Save fault names
with open(OUTPUT_DIR / 'fault_column_names.pkl', 'wb') as f:
    pickle.dump(FAULT_COLUMNS, f)

print(f"\n" + "="*80)
print("COMPLETE! You can now run analysis steps 04-10 with the full dataset.")
print("="*80)
