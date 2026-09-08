#!/usr/bin/env python3
"""Extract SHAP values for Classifier Chain model (Phase B) and save npz.

Loads the saved CC model from the cooked data directory, loads features (X_scaled and feature names),
computes SHAP values per label estimator using TreeExplainer when possible, and writes a .npz file
with arrays 'shap_0', 'shap_1', ... plus 'label_names' and 'feature_names'.
"""
import logging
from pathlib import Path
import pickle
import numpy as np
import shap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('extract_shap_cc')

import os
COOKED = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data'))
PHASEB = COOKED / 'step_08_phaseB_cc'
MODEL_PKL = PHASEB / 'cc_model.pkl'
OUT_NPZ = PHASEB / 'shap_cc.npz'

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
    # cc_model.pkl is now {'model': cc, 'scaler': ..., 'pca': ..., 'feature_cols': ...}
    # (leakage-safe phaseB_cc_test.py). Fall back to the old bare-model format
    # for any pkl produced before that fix.
    if isinstance(cc_data, dict) and 'model' in cc_data:
        cc = cc_data['model']
        scaler, pca, feature_names = cc_data.get('scaler'), cc_data.get('pca'), cc_data.get('feature_cols')
    else:
        cc, scaler, pca, feature_names = cc_data, None, None, None

    pkl_data = load_features()
    if scaler is not None and feature_names is not None:
        # Transform with the SAME train-fold-fit scaler/PCA the model was
        # trained on -- not the globally pre-fit X_scaled (see
        # leakage_safe_features.py for why that matters).
        X_raw = pkl_data['features_all'][feature_names].fillna(0).values
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e10, neginf=-1e10)
        X = scaler.transform(X_raw)
        if pca is not None:
            X = pca.transform(X)
    else:
        logger.warning('cc_model.pkl has no saved scaler (old-format pkl) -- '
                        'falling back to the globally pre-fit X_scaled')
        X = pkl_data.get('X_scaled')
        feature_names = pkl_data.get('feature_names') or pkl_data.get('feature_cols')
    if X is None:
        logger.error('Feature data X is None')
        raise SystemExit(1)

    # For speed: compute SHAP on a representative sample if dataset is large
    max_shap_samples = 500
    if X.shape[0] > max_shap_samples:
        logger.info('Sampling %d rows for SHAP computation (dataset has %d rows)', max_shap_samples, X.shape[0])
        idx = np.random.choice(X.shape[0], max_shap_samples, replace=False)
        X_shap = X[idx]
    else:
        X_shap = X

    # label names: if model has attribute classes_ or estimators_
    n_labels = len(getattr(cc, 'estimators_', []))
    # ClassifierChain.estimators_ is ordered by the chain's fit order (cc.order_),
    # not necessarily the original label order -- use it so label_names reflects
    # which original label each position actually predicts.
    order = list(getattr(cc, 'order_', range(n_labels)))
    label_names = [f'label_{order[pos]}' for pos in range(n_labels)]

    save_dict = {'label_names': np.array(label_names), 'feature_names': np.array(feature_names)}

    logger.info('Computing SHAP for %d labels (this may take time)', n_labels)
    # ClassifierChain feeds each subsequent estimator the BASE features plus
    # every previously-chained label's prediction as extra columns -- position
    # p in the chain expects n_features + p columns, not a fixed width (found
    # 2026-09-05: every position past 0 was silently failing both TreeExplainer
    # and KernelExplainer with a "feature shape mismatch" because this script
    # fed every estimator the same fixed-width X_shap). Reconstruct the same
    # progressively-widened input ClassifierChain.predict() itself would feed
    # forward, growing X_aug by one column after each position using that
    # position's own prediction.
    X_aug = X_shap
    for pos, est in enumerate(cc.estimators_):
        key = f'shap_{pos}'
        try:
            expl = shap.TreeExplainer(est)
            sv = expl.shap_values(X_aug)
            save_dict[key] = np.array(sv, dtype=object)
            logger.info('Computed Tree SHAP for label %d (chain position %d, %d input cols)',
                        order[pos], pos, X_aug.shape[1])
        except Exception as e:
            logger.warning('TreeExplainer failed for label %d: %s — falling back to KernelExplainer (slow)',
                            order[pos], e)
            try:
                # Use a small background of 50 samples for KernelExplainer
                background = X_aug[np.random.choice(X_aug.shape[0], min(30, X_aug.shape[0]), replace=False)]
                expl = shap.KernelExplainer(lambda v: est.predict_proba(v)[:,1] if hasattr(est, 'predict_proba') else est.predict(v), background)
                sv = expl.shap_values(X_aug, nsamples=100)
                save_dict[key] = np.array(sv, dtype=object)
                logger.info('Computed Kernel SHAP for label %d', order[pos])
            except Exception as e2:
                logger.exception('KernelExplainer also failed for label %d: %s', order[pos], e2)

        # Append this position's own prediction as the extra column the NEXT
        # estimator in the chain expects. Guarded separately so a prediction
        # failure here can't take down the remaining positions' SHAP values.
        try:
            pred_this = np.asarray(est.predict(X_aug)).reshape(-1, 1)
        except Exception as e3:
            logger.warning('predict() failed for label %d while extending the chain input: %s '
                            '-- padding with zeros', order[pos], e3)
            pred_this = np.zeros((X_aug.shape[0], 1))
        X_aug = np.hstack([X_aug, pred_this])

    np.savez_compressed(OUT_NPZ, **save_dict)
    logger.info('Saved SHAP archive to %s', OUT_NPZ)

if __name__ == '__main__':
    main()
