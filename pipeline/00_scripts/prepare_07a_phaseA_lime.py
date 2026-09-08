#!/usr/bin/env python3
"""
Phase A - LIME Value Extraction for Binary Relevance Models
=============================================================
Computes LIME local explanations for each label's Random Forest classifier
and saves them alongside the existing SHAP output (prepare_07a_phaseA_shap.py).

Added 2026-09-05: LIME was named in the pipeline description (see
pipeline/diagrams/) but not implemented anywhere in the codebase; only SHAP
existed, for 3 of the multi-label models. This closes that gap for Phase A.

LIME explains one instance at a time (fits a local linear surrogate via
perturbation sampling), which is much more expensive per-instance than SHAP's
TreeExplainer -- so this uses a smaller sample size and only keeps each
instance's top-K contributing features (LIME's native, sparse output), rather
than forcing a SHAP-style dense per-feature array.

Input:
  - br_results.pkl (trained BR model)
  - features_engineered.pkl (X data)

Output:
  - lime_br.npz (per-label, per-sampled-instance top-K feature attributions)
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
OUTPUT_FILE = PHASEA_DIR / 'lime_br.npz'

# LIME is far more expensive per-instance than SHAP's TreeExplainer (it fits a
# local perturbation-sampled surrogate model per instance), so these are kept
# small relative to SHAP_SAMPLE_SIZE=500 elsewhere in this codebase.
LIME_SAMPLE_SIZE = 30
LIME_NUM_FEATURES = 20   # top-K contributing features kept per explanation
LIME_NUM_PERTURBATIONS = 500  # LIME's internal num_samples for the local fit


def main():
    logger.info("="*70)
    logger.info("PHASE A: LIME VALUE EXTRACTION")
    logger.info("="*70)

    logger.info(f"Loading features from {FEATURES_FILE}")
    with open(FEATURES_FILE, 'rb') as f:
        data = pickle.load(f)

    y = data.get('y_multilabel')

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
    feature_names = list(feature_cols) if pca is None else [f'pc_{i}' for i in range(X.shape[1])]

    logger.info(f"Data shape: X={X.shape}, y={y.shape}")

    n_labels = y.shape[1]
    logger.info(f"Number of labels: {n_labels}")

    n_samples = min(LIME_SAMPLE_SIZE, X.shape[0])
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(X.shape[0], size=n_samples, replace=False)
    X_sample = X[sample_idx]

    logger.info(f"Using {n_samples} samples for LIME computation "
                f"({LIME_NUM_FEATURES} top features/instance, {LIME_NUM_PERTURBATIONS} perturbations/instance)")

    try:
        from lime.lime_tabular import LimeTabularExplainer
        import lime
        logger.info(f"LIME version: {getattr(lime, '__version__', 'unknown')}")
    except ImportError:
        logger.error("lime not installed. Install with: pip install lime")
        return 1

    # Background/training reference distribution for LIME's perturbation
    # sampling: a random subset of the full (post-scaler/PCA) feature space,
    # same convention as the SHAP background sample elsewhere in this codebase.
    bg_idx = rng.choice(X.shape[0], size=min(2000, X.shape[0]), replace=False)
    explainer = LimeTabularExplainer(
        training_data=X[bg_idx],
        feature_names=feature_names,
        class_names=['Normal', 'Fault'],
        mode='classification',
        discretize_continuous=False,  # these are physics features, not natural bins
        random_state=42,
    )

    lime_dict = {}

    for i in range(n_labels):
        logger.info(f"Computing LIME for label {i+1}/{n_labels}...")
        clf = br_model.estimators_[i]

        # Dense (n_samples, n_features) array: LIME only returns its top-K
        # contributing features per instance, so every other column is 0 --
        # honest reflection of LIME's sparse, local-surrogate design (unlike
        # SHAP's dense per-feature attribution).
        label_matrix = np.zeros((n_samples, X.shape[1]), dtype=np.float32)

        for row, x in enumerate(X_sample):
            try:
                exp = explainer.explain_instance(
                    x, clf.predict_proba,
                    num_features=LIME_NUM_FEATURES,
                    num_samples=LIME_NUM_PERTURBATIONS,
                )
                for feat_idx, weight in exp.as_map()[1]:  # class 1 = Fault
                    label_matrix[row, feat_idx] = weight
            except Exception as e:
                logger.warning(f"  Label {i}, instance {row}: LIME failed - {e}")

        lime_dict[f'lime_{i}'] = label_matrix
        logger.info(f"  Label {i}: LIME matrix shape = {label_matrix.shape}")

    lime_dict['sample_idx'] = sample_idx
    lime_dict['feature_names'] = np.array(feature_names, dtype=object)

    logger.info(f"Saving LIME values to {OUTPUT_FILE}")
    np.savez_compressed(OUTPUT_FILE, **lime_dict)

    file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Saved: {OUTPUT_FILE} ({file_size:.2f} MB)")

    logger.info("="*70)
    logger.info("LIME EXTRACTION COMPLETE")
    logger.info("="*70)

    return 0


if __name__ == '__main__':
    exit(main())
