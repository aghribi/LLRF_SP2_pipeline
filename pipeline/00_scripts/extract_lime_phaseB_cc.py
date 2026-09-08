#!/usr/bin/env python3
"""Extract LIME explanations for Classifier Chain model (Phase B) and save npz.

Added 2026-09-05: LIME was named in the pipeline description but not
implemented anywhere; only SHAP existed (see extract_shap_phaseB_cc.py, which
this mirrors, including the ClassifierChain chain-widening fix from the same
date -- position p in the chain expects n_features + p columns since
ClassifierChain appends each prior label's own prediction as an extra input
column for the next estimator).

LIME explains one instance at a time (local perturbation-sampled surrogate),
much more expensive per-instance than SHAP's TreeExplainer, so this keeps a
smaller sample size and only keeps each instance's top-K contributing
features (LIME's native sparse output) rather than a dense SHAP-style array.
"""
import logging
from pathlib import Path
import pickle
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('extract_lime_cc')

import os
COOKED = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data'))
PHASEB = COOKED / 'step_08_phaseB_cc'
MODEL_PKL = PHASEB / 'cc_model.pkl'
OUT_NPZ = PHASEB / 'lime_cc.npz'

LIME_SAMPLE_SIZE = 30
LIME_NUM_FEATURES = 20
LIME_NUM_PERTURBATIONS = 500


def load_features():
    candidates = [COOKED / 'step_03_features' / 'features_engineered.pkl', COOKED / 'features_engineered.pkl']
    for p in candidates:
        if p.exists():
            logger.info('Loading features from %s', p)
            with open(p, 'rb') as f:
                obj = pickle.load(f)
            return obj
    logger.error('No features artifact found in expected locations')
    raise SystemExit(1)


def main():
    if not MODEL_PKL.exists():
        logger.error('CC model not found: %s', MODEL_PKL)
        raise SystemExit(1)
    with open(MODEL_PKL, 'rb') as f:
        cc_data = pickle.load(f)
    if isinstance(cc_data, dict) and 'model' in cc_data:
        cc = cc_data['model']
        scaler, pca, feature_names = cc_data.get('scaler'), cc_data.get('pca'), cc_data.get('feature_cols')
    else:
        cc, scaler, pca, feature_names = cc_data, None, None, None

    pkl_data = load_features()
    if scaler is not None and feature_names is not None:
        X_raw = pkl_data['features_all'][feature_names].fillna(0).values
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e10, neginf=-1e10)
        X = scaler.transform(X_raw)
        if pca is not None:
            X = pca.transform(X)
            feature_names = [f'pc_{i}' for i in range(X.shape[1])]
    else:
        logger.warning('cc_model.pkl has no saved scaler (old-format pkl) -- '
                        'falling back to the globally pre-fit X_scaled')
        X = pkl_data.get('X_scaled')
        feature_names = pkl_data.get('feature_names') or pkl_data.get('feature_cols')
    if X is None:
        logger.error('Feature data X is None')
        raise SystemExit(1)
    feature_names = list(feature_names)

    try:
        from lime.lime_tabular import LimeTabularExplainer
        import lime
        logger.info('LIME version: %s', getattr(lime, '__version__', 'unknown'))
    except ImportError:
        logger.error('lime not installed. Install with: pip install lime')
        raise SystemExit(1)

    n_samples = min(LIME_SAMPLE_SIZE, X.shape[0])
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(X.shape[0], size=n_samples, replace=False)
    X_sample = X[sample_idx]

    n_labels = len(getattr(cc, 'estimators_', []))
    order = list(getattr(cc, 'order_', range(n_labels)))
    label_names = [f'label_{order[pos]}' for pos in range(n_labels)]

    save_dict = {'label_names': np.array(label_names), 'feature_names': np.array(feature_names)}

    logger.info('Computing LIME for %d labels (%d samples, this may take a while)', n_labels, n_samples)

    # Same progressive chain-widening as extract_shap_phaseB_cc.py: build the
    # augmented background/explain matrices position by position, appending
    # each position's own prediction before moving to the next.
    bg_idx = rng.choice(X.shape[0], size=min(2000, X.shape[0]), replace=False)
    X_bg_aug = X[bg_idx]
    X_aug = X_sample
    aug_feature_names = list(feature_names)

    for pos, est in enumerate(cc.estimators_):
        key = f'lime_{pos}'
        label_matrix = np.zeros((n_samples, len(feature_names)), dtype=np.float32)

        try:
            explainer = LimeTabularExplainer(
                training_data=X_bg_aug,
                feature_names=aug_feature_names,
                class_names=['Normal', 'Fault'],
                mode='classification',
                discretize_continuous=False,
                random_state=42,
            )
            for row, x in enumerate(X_aug):
                try:
                    exp = explainer.explain_instance(
                        x, est.predict_proba,
                        num_features=LIME_NUM_FEATURES,
                        num_samples=LIME_NUM_PERTURBATIONS,
                    )
                    for feat_idx, weight in exp.as_map()[1]:
                        if feat_idx < len(feature_names):  # ignore chain-appended columns in the output
                            label_matrix[row, feat_idx] = weight
                except Exception as e:
                    logger.warning('  Label %d, instance %d: LIME failed - %s', order[pos], row, e)
            logger.info('Computed LIME for label %d (chain position %d, %d input cols)',
                        order[pos], pos, X_aug.shape[1])
        except Exception as e:
            logger.exception('LIME setup failed for label %d: %s', order[pos], e)

        save_dict[key] = label_matrix

        # Extend both the explain and background matrices with this
        # position's own prediction, same as the SHAP script.
        try:
            pred_this = np.asarray(est.predict(X_aug)).reshape(-1, 1)
            pred_bg = np.asarray(est.predict(X_bg_aug)).reshape(-1, 1)
        except Exception as e3:
            logger.warning('predict() failed for label %d while extending the chain input: %s '
                            '-- padding with zeros', order[pos], e3)
            pred_this = np.zeros((X_aug.shape[0], 1))
            pred_bg = np.zeros((X_bg_aug.shape[0], 1))
        X_aug = np.hstack([X_aug, pred_this])
        X_bg_aug = np.hstack([X_bg_aug, pred_bg])
        aug_feature_names = aug_feature_names + [f'_chain_label_{order[pos]}']

    save_dict['sample_idx'] = sample_idx
    np.savez_compressed(OUT_NPZ, **save_dict)
    logger.info('Saved LIME archive to %s', OUT_NPZ)


if __name__ == '__main__':
    main()
