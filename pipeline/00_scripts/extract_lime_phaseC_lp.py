#!/usr/bin/env python3
"""Extract LIME explanations for LabelPowerset model (Phase C) and save npz.

Added 2026-09-05: LIME was named in the pipeline description but not
implemented anywhere; only SHAP existed (see extract_shap_phaseC_lp.py, which
this mirrors).

Label Powerset predicts a single multi-class output (one class per observed
label-combination), not one binary output per label like BR/CC -- so unlike
the SHAP script (which asks for every class's attribution), this explains
each instance's own top-predicted class only (LIME's standard `top_labels=1`
usage for a multi-class model), since the number of powerset classes can be
large and instance-specific.
"""
import logging
from pathlib import Path
import pickle
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('extract_lime_lp')

import os
COOKED = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data'))
PHASEC = COOKED / 'step_09_phaseC_lp'
MODEL_PKL = PHASEC / 'lp_model.pkl'
OUT_NPZ = PHASEC / 'lime_lp.npz'

LIME_SAMPLE_SIZE = 30
LIME_NUM_FEATURES = 20
LIME_NUM_PERTURBATIONS = 500


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


def get_proba_fn(lp):
    """LabelPowersetWrapper has no predict_proba of its own -- its inner
    estimator (fit on integer-encoded powerset classes) does."""
    if hasattr(lp, 'predict_proba'):
        return lp.predict_proba
    if hasattr(lp, 'estimator') and hasattr(lp.estimator, 'predict_proba'):
        return lp.estimator.predict_proba
    raise ValueError('No predict_proba available on the LP model or its inner estimator')


def infer_class_tuple_names(lp, n_classes):
    inv = getattr(lp, 'inv_map', None) or getattr(getattr(lp, 'estimator', None), 'inv_map', None)
    if inv:
        return ['_'.join(map(str, inv.get(i, (i,)))) for i in range(n_classes)]
    return [f'class_{i}' for i in range(n_classes)]


def main():
    if not MODEL_PKL.exists():
        logger.error('LP model not found: %s', MODEL_PKL)
        raise SystemExit(1)
    with open(MODEL_PKL, 'rb') as f:
        lp_data = pickle.load(f)
    if isinstance(lp_data, dict) and 'model' in lp_data:
        lp = lp_data['model']
        scaler, pca, feature_names = lp_data.get('scaler'), lp_data.get('pca'), lp_data.get('feature_cols')
    else:
        lp, scaler, pca, feature_names = lp_data, None, None, None

    pkl_data = load_features()
    if scaler is not None and feature_names is not None:
        X_raw = pkl_data['features_all'][feature_names].fillna(0).values
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e10, neginf=-1e10)
        X = scaler.transform(X_raw)
        if pca is not None:
            X = pca.transform(X)
            feature_names = [f'pc_{i}' for i in range(X.shape[1])]
    else:
        logger.warning('lp_model.pkl has no saved scaler (old-format pkl) -- '
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

    proba_fn = get_proba_fn(lp)
    n_classes = proba_fn(X[:1]).shape[1]
    class_names = infer_class_tuple_names(lp, n_classes)
    logger.info('Label Powerset has %d classes', n_classes)

    n_samples = min(LIME_SAMPLE_SIZE, X.shape[0])
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(X.shape[0], size=n_samples, replace=False)
    X_sample = X[sample_idx]
    bg_idx = rng.choice(X.shape[0], size=min(2000, X.shape[0]), replace=False)

    explainer = LimeTabularExplainer(
        training_data=X[bg_idx],
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=False,
        random_state=42,
    )

    logger.info('Computing LIME for %d instances (top predicted class each, this may take a while)', n_samples)

    # Sparse (n_samples, n_features) matrix: only the instance's own
    # top-predicted powerset class is explained per row (LIME's top_labels=1),
    # since which class is "the" prediction varies instance to instance.
    lime_matrix = np.zeros((n_samples, len(feature_names)), dtype=np.float32)
    top_class_idx = np.full(n_samples, -1, dtype=np.int32)

    for row, x in enumerate(X_sample):
        try:
            exp = explainer.explain_instance(
                x, proba_fn,
                num_features=LIME_NUM_FEATURES,
                num_samples=LIME_NUM_PERTURBATIONS,
                top_labels=1,
            )
            predicted_label = exp.available_labels()[0]
            top_class_idx[row] = predicted_label
            for feat_idx, weight in exp.as_map()[predicted_label]:
                lime_matrix[row, feat_idx] = weight
        except Exception as e:
            logger.warning('  Instance %d: LIME failed - %s', row, e)

    save_dict = {
        'lime_top_class': lime_matrix,
        'top_class_idx': top_class_idx,
        'class_names': np.array(class_names, dtype=object),
        'feature_names': np.array(feature_names, dtype=object),
        'sample_idx': sample_idx,
    }
    np.savez_compressed(OUT_NPZ, **save_dict)
    logger.info('Saved LIME archive to %s', OUT_NPZ)


if __name__ == '__main__':
    main()
