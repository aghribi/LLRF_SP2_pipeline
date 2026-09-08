#!/usr/bin/env python3
"""Extract SHAP values for LabelPowerset model (Phase C) and save npz.

Loads the saved LP model from the cooked data directory, loads features (X_scaled and feature names),
computes SHAP values using TreeExplainer when possible, and writes a .npz file with keys
like 'shap_0', 'shap_1', ... plus 'label_names' and 'feature_names'.
"""
import logging
from pathlib import Path
import pickle
import numpy as np
import shap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('extract_shap_lp')

import os
COOKED = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data'))
PHASEC = COOKED / 'step_09_phaseC_lp'
MODEL_PKL = PHASEC / 'lp_model.pkl'
OUT_NPZ = PHASEC / 'shap_lp.npz'


# Provide a compatible LabelPowersetWrapper so pickle.loads can find the class
class LabelPowersetWrapper:
    def __init__(self, estimator):
        self.estimator = estimator
        self.mapping = {}
        self.inv_map = {}

    def fit(self, X, Y):
        tuples = [tuple(map(int, row)) for row in Y]
        uniq = {}
        curr = 0
        y_enc = []
        for t in tuples:
            if t not in uniq:
                uniq[t] = curr
                curr += 1
            y_enc.append(uniq[t])
        self.mapping = uniq
        self.inv_map = {v: k for k, v in uniq.items()}
        self.estimator.fit(X, np.array(y_enc))

    def predict(self, X):
        y_enc = self.estimator.predict(X)
        out = []
        for v in y_enc:
            tup = self.inv_map.get(int(v))
            if tup is None:
                out.append([0] * len(next(iter(self.mapping.keys()))))
            else:
                out.append(list(tup))
        return np.array(out)


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


def infer_label_names(model):
    # Try common attributes
    if hasattr(model, 'classes_'):
        try:
            return [str(c) for c in model.classes_]
        except Exception:
            pass
    # LabelPowerset wrapper: try inv_map
    inv = getattr(model, 'inv_map', None) or getattr(getattr(model, 'estimator', None), 'inv_map', None)
    if inv:
        # inv_map: int -> tuple
        names = []
        for i in range(max(inv.keys()) + 1):
            tup = inv.get(i)
            if tup is None:
                names.append(f'class_{i}')
            else:
                names.append('_'.join(map(str, tup)))
        return names
    # Fallback
    return ['label_0']


def save_shap_dict(save_dict):
    np.savez_compressed(OUT_NPZ, **save_dict)
    logger.info('Saved SHAP archive to %s', OUT_NPZ)


def main():
    if not MODEL_PKL.exists():
        logger.error('LP model not found: %s', MODEL_PKL)
        raise SystemExit(1)
    with open(MODEL_PKL, 'rb') as f:
        lp_data = pickle.load(f)
    # lp_model.pkl is now {'model': lp, 'scaler': ..., 'pca': ..., 'feature_cols': ...}
    # (leakage-safe phaseC_lp_test.py). Fall back to the old bare-model format
    # for any pkl produced before that fix.
    if isinstance(lp_data, dict) and 'model' in lp_data:
        lp = lp_data['model']
        scaler, pca, feature_names = lp_data.get('scaler'), lp_data.get('pca'), lp_data.get('feature_cols')
    else:
        lp, scaler, pca, feature_names = lp_data, None, None, None

    pkl_data = load_features()
    if scaler is not None and feature_names is not None:
        # Transform with the SAME train-fold-fit scaler/PCA the model was
        # trained on -- not the globally pre-fit X_scaled.
        X_raw = pkl_data['features_all'][feature_names].fillna(0).values
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e10, neginf=-1e10)
        X = scaler.transform(X_raw)
        if pca is not None:
            X = pca.transform(X)
    else:
        logger.warning('lp_model.pkl has no saved scaler (old-format pkl) -- '
                        'falling back to the globally pre-fit X_scaled')
        X = pkl_data.get('X_scaled')
        feature_names = pkl_data.get('feature_names') or pkl_data.get('feature_cols')
    if X is None:
        logger.error('Feature data X is None')
        raise SystemExit(1)

    label_names = infer_label_names(lp)
    save_dict = {'label_names': np.array(label_names), 'feature_names': np.array(feature_names)}

    logger.info('Computing SHAP (this may take time)')
    try:
        expl = shap.TreeExplainer(lp)
        sv = expl.shap_values(X)
    except Exception as e:
        logger.warning('TreeExplainer failed: %s — falling back to KernelExplainer (slow)', e)
        try:
            background = X[np.random.choice(X.shape[0], min(50, X.shape[0]), replace=False)]
            expl = shap.KernelExplainer(lambda v: lp.predict_proba(v) if hasattr(lp, 'predict_proba') else lp.predict(v), background)
            sv = expl.shap_values(X, nsamples=100)
        except Exception as e2:
            logger.exception('KernelExplainer failed: %s', e2)
            raise SystemExit(1)

    # Normalize different return shapes into shap_{i} entries
    try:
        if isinstance(sv, list):
            for i, arr in enumerate(sv):
                save_dict[f'shap_{i}'] = np.array(arr, dtype=object)
        else:
            arr = np.asarray(sv)
            if arr.ndim == 3 and arr.shape[0] == len(label_names):
                # (n_classes, n_samples, n_features)
                for i in range(arr.shape[0]):
                    save_dict[f'shap_{i}'] = arr[i]
            elif arr.ndim == 3 and arr.shape[-1] == len(label_names):
                # (n_samples, n_features, n_classes)
                for i in range(arr.shape[-1]):
                    save_dict[f'shap_{i}'] = arr[..., i]
            else:
                # single 2D array: save as shap_0
                save_dict['shap_0'] = arr
    except Exception:
        logger.exception('Failed to normalize SHAP values; saving raw object')
        save_dict['shap_0'] = np.array(sv, dtype=object)

    save_shap_dict(save_dict)


if __name__ == '__main__':
    main()
