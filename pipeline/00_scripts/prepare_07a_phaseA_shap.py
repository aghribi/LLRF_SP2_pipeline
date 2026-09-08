#!/usr/bin/env python3
"""
Phase A - SHAP Value Extraction for Binary Relevance Models
============================================================
Computes SHAP values for each label's Random Forest classifier
and saves them for the analysis notebook.

Input:
  - br_results.pkl (trained BR model)
  - features_engineered.pkl (X data)

Output:
  - shap_br.npz (SHAP values for each label)
"""

import pickle
from pathlib import Path
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
import os
COOKED = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data'))
PHASEA_DIR = COOKED / 'step_07_phaseA_br'
FEATURES_FILE = COOKED / 'features_engineered.pkl'
BR_FILE = PHASEA_DIR / 'br_results.pkl'
OUTPUT_FILE = PHASEA_DIR / 'shap_br.npz'

# SHAP sample size (to keep computation tractable)
SHAP_SAMPLE_SIZE = 500


def main():
    logger.info("="*70)
    logger.info("PHASE A: SHAP VALUE EXTRACTION")
    logger.info("="*70)

    # Load features
    logger.info(f"Loading features from {FEATURES_FILE}")
    with open(FEATURES_FILE, 'rb') as f:
        data = pickle.load(f)

    y = data.get('y_multilabel')

    # Load BR model (+ the scaler/PCA that were fit on ITS train fold only --
    # see leakage_safe_features.py. We must transform with the SAME
    # train-fold-fit scaler/PCA the model was trained on, not the globally
    # pre-fit X_scaled, or the SHAP explanations don't match what the model
    # actually saw.)
    logger.info(f"Loading BR model from {BR_FILE}")
    with open(BR_FILE, 'rb') as f:
        br_data = pickle.load(f)

    br_model = br_data.get('model')
    if br_model is None:
        logger.error("No model found in br_results.pkl")
        return 1

    scaler = br_data.get('scaler')
    pca = br_data.get('pca')
    feature_cols = br_data.get('feature_cols')
    if scaler is None or feature_cols is None:
        logger.error("br_results.pkl missing scaler/feature_cols -- retrain with the "
                      "leakage-safe phaseA_br_test.py first")
        return 1

    X_raw = data['features_all'][feature_cols].fillna(0).values
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e10, neginf=-1e10)
    X = scaler.transform(X_raw)
    if pca is not None:
        X = pca.transform(X)
    feature_names = feature_cols

    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Features: {len(feature_names) if feature_names else 'unknown'}")

    # Get number of labels
    n_labels = y.shape[1]
    logger.info(f"Number of labels: {n_labels}")

    # Sample data for SHAP computation
    n_samples = min(SHAP_SAMPLE_SIZE, X.shape[0])
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(X.shape[0], size=n_samples, replace=False)
    X_sample = X[sample_idx]

    logger.info(f"Using {n_samples} samples for SHAP computation")

    # Import SHAP
    try:
        import shap
        logger.info(f"SHAP version: {shap.__version__}")
    except ImportError:
        logger.error("SHAP not installed. Install with: pip install shap")
        return 1

    # Compute SHAP values for each label
    shap_dict = {}

    for i in range(n_labels):
        logger.info(f"Computing SHAP for label {i+1}/{n_labels}...")

        # Get the classifier for this label
        clf = br_model.estimators_[i]

        try:
            # Use TreeExplainer for Random Forest
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_sample)

            # For binary classification, shap_values is [neg_class, pos_class]
            # We want positive class contribution
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class

            shap_dict[f'shap_{i}'] = shap_values
            logger.info(f"  Label {i}: SHAP shape = {shap_values.shape}")

        except Exception as e:
            logger.warning(f"  Label {i}: SHAP failed - {e}")
            # Store zeros as placeholder
            shap_dict[f'shap_{i}'] = np.zeros((n_samples, X.shape[1]))

    # Add metadata
    shap_dict['sample_idx'] = sample_idx
    if feature_names:
        shap_dict['feature_names'] = np.array(feature_names, dtype=object)

    # Save SHAP values
    logger.info(f"Saving SHAP values to {OUTPUT_FILE}")
    np.savez_compressed(OUTPUT_FILE, **shap_dict)

    # Verify file
    file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Saved: {OUTPUT_FILE} ({file_size:.2f} MB)")

    logger.info("="*70)
    logger.info("SHAP EXTRACTION COMPLETE")
    logger.info("="*70)

    return 0


if __name__ == '__main__':
    exit(main())
