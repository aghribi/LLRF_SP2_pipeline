"""
Leakage-safe train/test feature builder for SPIRAL2 LLRF modeling scripts.

Background: prepare_data_cluster*.py fits StandardScaler + PCA on the FULL
dataset (train+test combined) before any downstream script ever splits the
data, then saves the result as X_scaled/X_pca in features_engineered*.pkl.
Every step-04-10 / Phase A-D script that trained/evaluated a model by
consuming X_scaled/X_pca directly was therefore training and testing on a
scaling/PCA transform partly informed by its own held-out test set (found
2026-09-04 alongside the interlock_type label-leak investigation; see
NOTEBOOK_AUDIT_2026-09-03.md and the interlock_type fix in
prepare_data_cluster_v6.py).

This helper rebuilds X from the pkl's raw, unscaled features_all[feature_cols]
and fits scaler/PCA on the training fold only, so no downstream script needs
to reimplement this by hand. Test (and, if requested, validation) folds are
transformed using the training-fold-fitted scaler/PCA -- never refit on them.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


GENUINE_PRECURSOR_SUFFIXES = ['_trend_slope', '_trend_acceleration', '_cusum_max', '_cusum_min',
                              '_early_late_diff', '_early_late_ratio', '_var_ratio_late_early']


def true_precursor_columns(feature_cols):
    """
    The subset of feature_cols that is safe for a genuine "before this event's
    own outcome is decided" precursor task -- built on a FIXED physical-
    duration window immediately before each event's own trigger index
    (engineer_features() sections 4 and 6 in prepare_data_cluster_v6.py), with
    a `continue`-if-too-short guard against variable pre-trigger window
    length. Everything else in feature_cols is one of: Section 1's
    whole-signal statistics (no pre/post split at all -- includes post-trigger
    content unconditionally), Section 2's "steady-state" pre/post features
    (deliberately use ALL available pre-trigger data, unbounded -- confirmed
    2026-09-06 to be confounded by a systematic pre-trigger-window-length
    difference between Normal and Fault event captures, see
    investigate_precursor_window_leak.py), or explicit post-trigger features.

    2026-09-06 investigation: restricting to this subset and re-running the
    supervised precursor RF still gives AUC=0.9998+/-0.0002 (10 repeated
    splits) while Isolation Forest/LOF on the SAME columns stay at 0.65/0.59
    -- i.e. this is genuine, structured signal, not a window-length artifact
    (window length is already fixed in this subset by construction). The
    remaining interpretive caveat: this window sits immediately before the
    *informatic* trigger, which can lag true physical fault onset -- so this
    should be reported as "early fault-onset detection ahead of the
    interlock trip," not "precursor detection" in the sense of before any
    physical change begins. See PHASE3_WEAKNESS_DIAGNOSIS_AND_OPERATIONAL_USABILITY.md.
    """
    named = [c for c in feature_cols if any(c.endswith(sfx) for sfx in GENUINE_PRECURSOR_SUFFIXES)]
    segments = [c for c in feature_cols if '_segment_' in c]
    return sorted(set(named) | set(segments))


def compute_scale_pos_weight(y_train):
    """
    Data-driven scale_pos_weight for XGBoost, computed from y_train's own
    class counts (aggregate negative/positive ratio across all label columns
    if 2D/multi-label, or the single column if 1D binary).

    Limitation, documented rather than hidden: unlike sklearn's
    class_weight='balanced' (which MultiOutputClassifier/ClassifierChain
    correctly recompute PER LABEL because they clone the base estimator and
    re-fit each clone on its own column), XGBoost's scale_pos_weight is a
    fixed hyperparameter baked into the estimator BEFORE ClassifierChain/
    LabelPowerset clones it, so the same value applies uniformly across every
    label position in the chain -- there is no per-label equivalent through
    these wrapper APIs without bypassing them with a manual per-column fit
    loop. This aggregate value is a reasonable, data-driven approximation,
    not a true per-label weight -- see Phase 4 of
    PHASE3_WEAKNESS_DIAGNOSIS_AND_OPERATIONAL_USABILITY.md's companion plan.
    """
    y_train = np.asarray(y_train)
    n_pos = float(np.sum(y_train == 1))
    n_neg = float(np.sum(y_train == 0))
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def raw_feature_matrix(pkl_data):
    """The unscaled, correlation-filtered feature matrix -- same columns X_scaled
    was built from, before StandardScaler/PCA were fit on the full dataset."""
    feature_cols = pkl_data['feature_cols']
    X_raw = pkl_data['features_all'][feature_cols].fillna(0).values
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e10, neginf=-1e10)
    return X_raw, feature_cols


def leakage_safe_split(pkl_data, y, test_size=0.3, val_size=None,
                        random_state=42, stratify=True,
                        use_pca=False, pca_variance=0.95):
    """
    Build train/(val/)test folds with scaler+PCA fit ONLY on the training fold.

    pkl_data : the loaded features_engineered*.pkl dict.
    y        : label array, same row order as pkl_data['features_all'].
    val_size : if given, carved out of the train fold (fraction of the
               original dataset, e.g. 0.1 for a 70/10/20 train/val/test split
               when test_size=0.2).
    use_pca  : if True, additionally PCA-reduces the scaled features (fit on
               the train fold only). Default False, matching every current
               step-04-10/Phase-A-D script's original behavior: they all
               consumed the pkl's unreduced 'X_scaled' (named, interpretable
               physics features), never 'X_pca' -- PCA is only used by the
               clustering scripts (09/09b/09c), which don't call this helper
               since they're unsupervised (no train/test split to protect).

    Returns a dict: X_train, X_test, y_train, y_test, idx_train, idx_test,
    scaler, pca (None if use_pca=False), feature_cols, and (if val_size is
    set) X_val, y_val, idx_val too.
    """
    X_raw, feature_cols = raw_feature_matrix(pkl_data)
    y = np.asarray(y)
    idx_all = np.arange(len(y))

    strat = y if stratify else None
    X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
        X_raw, y, idx_all, test_size=test_size, random_state=random_state, stratify=strat
    )

    X_val_raw = y_val = idx_val = None
    if val_size is not None:
        val_frac_of_train = val_size / (1.0 - test_size)
        strat_train = y_train if stratify else None
        X_train_raw, X_val_raw, y_train, y_val, idx_train, idx_val = train_test_split(
            X_train_raw, y_train, idx_train, test_size=val_frac_of_train,
            random_state=random_state, stratify=strat_train
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    X_val = scaler.transform(X_val_raw) if X_val_raw is not None else None

    pca = None
    if use_pca:
        pca = PCA(n_components=pca_variance, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        if X_val is not None:
            X_val = pca.transform(X_val)

    result = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'idx_train': idx_train, 'idx_test': idx_test,
        'scaler': scaler, 'pca': pca, 'feature_cols': feature_cols,
    }
    if X_val is not None:
        result.update({'X_val': X_val, 'y_val': y_val, 'idx_val': idx_val})
    return result


def repeated_leakage_safe_eval(pkl_data, y, model_fn, metric_fns, n_repeats=10,
                                test_size=0.3, stratify=False, base_random_state=0,
                                label_names=None, use_pca=False, pca_variance=0.95):
    """
    Repeat leakage_safe_split + fit + evaluate n_repeats times, each with a
    different random_state, to tell a genuine effect apart from single-split
    noise -- the 2026-09-05 audit found every current headline number traces
    to exactly one fixed-seed 70/30 split, so a rare category's reported F1
    (e.g. "Coupure externe rapide", 11 events total) has no error bar at all.

    model_fn(X_train, y_train) -> a fitted estimator exposing .predict(X_test).
    metric_fns : {metric_name: fn(y_test, y_pred) -> float OR a 1D array of
                  per-label scores (e.g. sklearn's f1_score(..., average=None)
                  for the multilabel case)}.
    label_names: names for a per-label metric's columns (falls back to
                 'label_0', 'label_1', ... if a metric returns an array and
                 this is None).

    Returns {metric_name: {'mean','std','min','max','values'}} for a scalar
    metric, or {metric_name: {'per_label': {label_name: {'mean','std','min',
    'max','values'}}}} for a metric whose fn returned a per-label array.
    """
    raw_values = {name: [] for name in metric_fns}
    for i in range(n_repeats):
        split = leakage_safe_split(pkl_data, y, test_size=test_size,
                                    random_state=base_random_state + i,
                                    stratify=stratify, use_pca=use_pca,
                                    pca_variance=pca_variance)
        model = model_fn(split['X_train'], split['y_train'])
        y_pred = model.predict(split['X_test'])
        for name, fn in metric_fns.items():
            raw_values[name].append(fn(split['y_test'], y_pred))

    summary = {}
    for name, values in raw_values.items():
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            summary[name] = {
                'mean': float(np.mean(arr)), 'std': float(np.std(arr)),
                'min': float(np.min(arr)), 'max': float(np.max(arr)),
                'values': [float(v) for v in arr],
            }
        else:
            n_labels = arr.shape[1]
            names = label_names if label_names is not None else [f'label_{j}' for j in range(n_labels)]
            per_label = {}
            for j, lname in enumerate(names):
                col = arr[:, j]
                per_label[lname] = {
                    'mean': float(np.mean(col)), 'std': float(np.std(col)),
                    'min': float(np.min(col)), 'max': float(np.max(col)),
                    'values': [float(v) for v in col],
                }
            summary[name] = {'per_label': per_label}
    return summary
