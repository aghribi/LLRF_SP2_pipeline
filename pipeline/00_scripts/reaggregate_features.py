#!/usr/bin/env python3
"""
Re-aggregate features from existing batches with updated physics feature protection.

This script:
1. Reads existing batch files (no re-extraction needed)
2. Adds control loop alias features
3. Applies correlation filtering with physics feature protection
4. Saves updated features_engineered.pkl

Usage:
    python reaggregate_features.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    OUTPUT_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')
    BATCH_DIR = OUTPUT_DIR / 'batches'

    logger.info("=" * 80)
    logger.info("RE-AGGREGATING FEATURES WITH PHYSICS PROTECTION")
    logger.info("=" * 80)

    # Load all batches
    batch_files = sorted(BATCH_DIR.glob('batch_*.pkl'))
    logger.info(f"Found {len(batch_files)} batch files")

    metadata_list = []
    features_list = []
    fault_labels = []
    signal_names = None

    for i, batch_file in enumerate(batch_files):
        logger.info(f"  [{i+1}/{len(batch_files)}] Processing {batch_file.name}...")
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)

        results = batch_data['results']
        for result in results:
            metadata_list.append(result['metadata'])
            features_list.append(result['features'])
            fault_labels.append(result['fault_labels'])

        if signal_names is None and results:
            signal_names = results[0].get('signal_names', [])

    logger.info(f"✓ Collected {len(features_list)} events from {len(batch_files)} batches")

    # Create DataFrame
    logger.info("\nCreating feature matrices and labels...")
    df_features = pd.DataFrame(features_list)

    # Add control loop alias features
    logger.info("Adding control loop alias features...")
    alias_mappings = {
        'amp_rms_pre': 'rf_mismatch_rms_pre',
        'amp_rms_post': 'rf_mismatch_rms_post',
        'phase_rms_pre': 'phase_std_pre',
        'phase_rms_post': 'phase_std_post',
        'control_overshoot_amp': 'rf_mismatch_max_excursion',
        'control_settling_time': 'rf_mismatch_settling_time',
        'control_saturation_fraction': 'reflected_saturation_fraction',
    }
    for alias_name, source_name in alias_mappings.items():
        if source_name in df_features.columns and alias_name not in df_features.columns:
            df_features[alias_name] = df_features[source_name]
            logger.info(f"  Added alias: {alias_name} <- {source_name}")

    logger.info(f"  Total features after aliases: {len(df_features.columns)}")

    # Create labels
    y_binary = np.array([f['binary'] for f in fault_labels])
    y_multilabel = np.array([f['multilabel'] for f in fault_labels])

    n_normal = np.sum(y_binary == 0)
    n_fault = np.sum(y_binary == 1)
    logger.info(f"\nDataset composition:")
    logger.info(f"  Total events: {len(y_binary)}")
    logger.info(f"  Normal: {n_normal} ({100*n_normal/len(y_binary):.1f}%)")
    logger.info(f"  Fault: {n_fault} ({100*n_fault/len(y_binary):.1f}%)")

    # Correlation filtering with physics protection
    logger.info("\nApplying feature selection (correlation filtering)...")
    feature_cols = df_features.columns.tolist()
    X_features = df_features.fillna(0).values

    # Define physics features to protect
    PROTECTED_PHYSICS_FEATURES = {
        # Effective Decay
        'effective_decay_tau', 'decay_rate', 'decay_fit_r2',
        'decay_relative_to_nominal', 'decay_non_exponentiality',
        # Detuning
        'detuning_mean_hz', 'detuning_std_hz', 'detuning_slope_hz_s',
        'detuning_peak_to_peak', 'detuning_jump_at_trigger',
        # Microphonics
        'microphonics_rms', 'microphonics_dom_freq',
        'microphonics_band_power_10_100', 'microphonics_q_factor',
        # Phase Stability
        'phase_jitter_rms', 'phase_excursion_pp',
        'phase_noise_slope', 'phase_psd_integral',
        # RF Power Flow
        'forward_power_mean', 'reflected_power_mean', 'power_reflection_ratio',
        'power_jump_at_trigger', 'power_imbalance_rms',
        # Control Loop (new aliases)
        'amp_rms_pre', 'amp_rms_post', 'phase_rms_pre', 'phase_rms_post',
        'control_overshoot_amp', 'control_settling_time', 'control_saturation_fraction',
        # RF Mismatch (original names)
        'rf_mismatch_rms_pre', 'rf_mismatch_rms_post', 'rf_mismatch_max_excursion',
        'rf_mismatch_settling_time', 'reflected_saturation_fraction',
        'phase_std_pre', 'phase_std_post',
        # Validity Flags ('beam_present': V6 and earlier; 'feed_forward_enabled': V7+,
        # 2026-09-09 rename of the same underlying header field -- see
        # prepare_data_cluster_v7.py's module docstring)
        'rf_drive_on', 'llrf_loop_closed', 'beam_present', 'feed_forward_enabled',
        'interlock_type', 'valid_decay_window',
        # Modulator features
        'mod_amplitude_mean_pre', 'mod_amplitude_std_pre', 'mod_phase_std_pre',
        'mod_amplitude_change', 'mod_phase_change', 'mod_amplitude_max_post',
    }

    corr_matrix = np.corrcoef(X_features.T)
    threshold = 0.95
    to_drop = set()

    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if abs(corr_matrix[i, j]) > threshold:
                # Only drop if NOT protected
                if feature_cols[j] not in PROTECTED_PHYSICS_FEATURES:
                    to_drop.add(feature_cols[j])

    protected_count = sum(1 for f in feature_cols if f in PROTECTED_PHYSICS_FEATURES)
    logger.info(f"  Protected physics features: {protected_count}")
    logger.info(f"  Features to drop (corr > {threshold}): {len(to_drop)}/{len(feature_cols)}")

    feature_cols_filtered = [col for col in feature_cols if col not in to_drop]
    df_features_filtered = df_features[feature_cols_filtered]
    X_filtered = df_features_filtered.fillna(0).values

    logger.info(f"  Remaining features after filtering: {len(feature_cols_filtered)}")

    # Clean data
    n_inf = np.isinf(X_filtered).sum()
    n_nan = np.isnan(X_filtered).sum()
    if n_inf > 0 or n_nan > 0:
        logger.warning(f"Found {n_inf} inf values and {n_nan} nan values. Replacing...")
        X_filtered = np.nan_to_num(X_filtered, nan=0.0, posinf=1e10, neginf=-1e10)

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    # PCA
    n_samples, n_features = X_scaled.shape
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(f"\nPCA results:")
    logger.info(f"  Original features: {X_scaled.shape[1]}")
    logger.info(f"  PCA components: {X_pca.shape[1]}")
    logger.info(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    # Verify physics features are preserved
    physics_in_filtered = [f for f in feature_cols_filtered if f in PROTECTED_PHYSICS_FEATURES]
    logger.info(f"\nPhysics features preserved: {len(physics_in_filtered)}")
    for f in sorted(physics_in_filtered):
        logger.info(f"  ✓ {f}")

    # Save
    features_file = OUTPUT_DIR / 'features_engineered.pkl'
    logger.info(f"\nSaving to: {features_file}")

    with open(features_file, 'wb') as f:
        pickle.dump({
            'features_all': df_features,
            'feature_cols': feature_cols_filtered,
            'X_scaled': X_scaled,
            'X_pca': X_pca,
            'y_binary': y_binary,
            'y_multilabel': y_multilabel,
            'scaler': scaler,
            'pca': pca,
            'metadata': metadata_list,
            'signal_names': signal_names if signal_names else [],
            'fault_column_names': [
                'Pickup threshold',
                'Fast external cutoff',
                'RF authorization absent',
                'Vacuum threshold',
                'Cavity quench/breakdown',
                'RF safety threshold exceeded',
                'RF regulation out of tolerance'
            ],
        }, f)

    logger.info(f"✓ Saved: {features_file}")
    logger.info(f"  File size: {features_file.stat().st_size / (1024**2):.1f} MB")

    logger.info("\n" + "=" * 80)
    logger.info("RE-AGGREGATION COMPLETE")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
