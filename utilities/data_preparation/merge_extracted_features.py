#!/usr/bin/env python3
"""
Merge Extracted Features - Combine temp files into final dataset
=================================================================

Merges the successfully extracted feature files from batches 0-15.
This gives us 3,370 events (4.4x more than the original 761).

Usage:
    python merge_extracted_features.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc

TEMP_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data/temp_features')
OUTPUT_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')

print("="*80)
print("MERGING EXTRACTED FEATURES (16 batches)")
print("="*80)

# Find temp files
result_files = sorted(TEMP_DIR.glob('features_*.pkl'))
print(f"\nFound {len(result_files)} extracted feature files")

FAULT_COLUMNS = [
    'Seuil pick-up',
    'Coupure externe rapide',
    'Absence autorisation RF',
    'Seuil de vide',
    'Claquage ou quench cavité',
    'Dép seuil de sécurité RF',
    'Rég signal RF hors tolérance'
]

# Merge all extracted files
all_features = []
all_metadata = []
all_y_binary = []
all_y_multilabel = []

for result_file in tqdm(result_files, desc="Merging"):
    with open(result_file, 'rb') as f:
        result = pickle.load(f)

    all_features.extend(result['features'])
    all_metadata.extend(result['metadata'])
    all_y_binary.extend(result['y_binary'])
    all_y_multilabel.extend(result['y_multilabel'])

    del result
    gc.collect()

print(f"\n✓ Merged {len(all_features)} events total")
print(f"  Features per event: {len(all_features[0])}")

# Create DataFrames
print("\nCreating DataFrames...")
df_features = pd.DataFrame(all_features)
df_metadata = pd.DataFrame(all_metadata)

del all_features, all_metadata
gc.collect()

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
y_binary = np.array(all_y_binary)
Y_multilabel = np.array(all_y_multilabel)

del all_y_binary, all_y_multilabel
gc.collect()

# Handle missing values and infinities
print(f"\nCleaning feature data...")
n_missing = df_features.isnull().sum().sum()
n_inf = np.isinf(df_features.values).sum()
print(f"  Found {n_missing} missing values")
print(f"  Found {n_inf} infinity values")

if n_missing > 0:
    df_features = df_features.fillna(df_features.median())
    df_features = df_features.fillna(0)
    print(f"  ✓ Filled missing values")

if n_inf > 0:
    # Replace inf/-inf with large but finite values
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.fillna(df_features.median())
    df_features = df_features.fillna(0)
    print(f"  ✓ Replaced infinity values")

# Scale and PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print(f"\nScaling features ({df_features.shape})...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

print(f"Running PCA...")
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"  PCA: {X_scaled.shape[1]} features → {X_pca.shape[1]} components ({pca.explained_variance_ratio_.sum():.1%} variance)")

# Save final result
print(f"\nSaving final dataset...")
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
    'sequences_full': None,
    'sequences_pretrigger': None,
}

output_file = OUTPUT_DIR / 'features_engineered.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(feature_data, f, protocol=4)

print(f"\n✓ SUCCESS! Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / (1024**2):.1f} MB")
print(f"  Events: {len(y_binary)}")
print(f"  Normal: {np.sum(y_binary == 0)} ({np.sum(y_binary == 0)/len(y_binary):.1%})")
print(f"  Anomaly: {np.sum(y_binary == 1)} ({np.sum(y_binary == 1)/len(y_binary):.1%})")
print(f"  Features: {X_scaled.shape[1]}")
print(f"  PCA components: {X_pca.shape[1]}")

# Save additional files
np.save(OUTPUT_DIR / 'fault_labels_matrix.npy', Y_multilabel)
with open(OUTPUT_DIR / 'fault_column_names.pkl', 'wb') as f:
    pickle.dump(FAULT_COLUMNS, f)

print(f"\n{'='*80}")
print("COMPLETE!")
print(f"Dataset expanded from 761 → {len(y_binary)} events (4.4x improvement!)")
print("You can now re-run analysis steps 04-10.")
print(f"{'='*80}")
