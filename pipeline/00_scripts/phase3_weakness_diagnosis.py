#!/usr/bin/env python3
"""
Phase 3: Weakness diagnosis (results-reliability plan).

Four quantitative pieces, each tied to a specific number rather than a
qualitative impression:
  1. Imbalance-performance correlation: per-category macro-F1 (from Phase 2's
     stability manifests) vs. category sample size.
  2. Confusion analysis: root-cause (7-class) and physics-subtype (9-class,
     within "Reg signal RF hors tolerance") confusion matrices.
  3. Learning curves: binary (step 06) and root-cause (step 08) RF models --
     tells "not enough data" apart from "wrong model/features".
  4. Dimensionality check: n_features vs. n_train_rows ratio.

Everything is computed live from V6 + Phase 2's manifests and written to
analysis/results_manifest/10_phase3_weakness_diagnosis.yaml -- no hand-typed
numbers, per this project's convention.
"""
import sys
import logging
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utilities.reporting.manifest import save_manifest

sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import leakage_safe_split, raw_feature_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler('phase3_weakness_diagnosis.log'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

V6_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6')
MANIFEST_DIR = Path(__file__).resolve().parents[2] / 'analysis' / 'results_manifest'


def load_v6():
    with open(V6_DIR / 'features_engineered_v6.pkl', 'rb') as f:
        return pickle.load(f)


def imbalance_correlation():
    logger.info("=" * 78)
    logger.info("1. IMBALANCE-PERFORMANCE CORRELATION")
    logger.info("=" * 78)

    data = load_v6()
    label_names = data['fault_column_names']
    sizes = data['y_multilabel'].sum(axis=0)

    manifests = ['07_phase2_multilabel_stability', '07A_phaseA_br_cc_stability',
                 '08_phaseB_cc_stability', '09_phaseC_lp_stability']
    per_script_f1 = {}
    for phase in manifests:
        path = MANIFEST_DIR / f'{phase}.yaml'
        if not path.exists():
            continue
        d = yaml.safe_load(path.read_text())
        per_label = d['stability']['metrics']['per_label_f1']['per_label']
        per_script_f1[phase] = {name: per_label[name]['mean'] for name in label_names}

    mean_f1_by_label = {
        name: float(np.mean([per_script_f1[p][name] for p in per_script_f1]))
        for name in label_names
    }

    size_arr = np.array([sizes[i] for i in range(len(label_names))], dtype=float)
    f1_arr = np.array([mean_f1_by_label[n] for n in label_names], dtype=float)
    pearson_r = float(np.corrcoef(size_arr, f1_arr)[0, 1])
    log_size_arr = np.log10(size_arr)
    pearson_r_log = float(np.corrcoef(log_size_arr, f1_arr)[0, 1])

    logger.info(f"{'Category':<32} {'N events':>9} {'Mean F1 (across 4 models)':>28}")
    for name in label_names:
        logger.info(f"{name:<32} {int(sizes[list(label_names).index(name)]):>9} {mean_f1_by_label[name]:>28.3f}")
    logger.info(f"\nPearson r (size vs F1):          {pearson_r:.3f}")
    logger.info(f"Pearson r (log10(size) vs F1):   {pearson_r_log:.3f}")

    return {
        'per_category': {n: {'n_events': int(sizes[list(label_names).index(n)]),
                              'mean_f1': mean_f1_by_label[n]} for n in label_names},
        'pearson_r_size_vs_f1': pearson_r,
        'pearson_r_log_size_vs_f1': pearson_r_log,
        'n_models_averaged': len(per_script_f1),
    }


def confusion_analysis():
    logger.info("=" * 78)
    logger.info("2. CONFUSION ANALYSIS")
    logger.info("=" * 78)

    data = load_v6()
    label_names = list(data['fault_column_names'])

    # --- Root cause (7-class, argmax of y_multilabel among fault events) ---
    y_multilabel = data['y_multilabel']
    has_fault = y_multilabel.sum(axis=1) > 0
    fault_indices = np.argmax(y_multilabel, axis=1)
    fault_data = {
        'features_all': data['features_all'].loc[has_fault].reset_index(drop=True),
        'feature_cols': data['feature_cols'],
    }
    y_root = fault_indices[has_fault]
    split = leakage_safe_split(fault_data, y_root, test_size=0.3, random_state=42, stratify=False)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(split['X_train'], split['y_train'])
    y_pred = clf.predict(split['X_test'])
    cm = confusion_matrix(split['y_test'], y_pred, labels=list(range(len(label_names))))

    logger.info("\nRoot-cause confusion matrix (rows=true, cols=predicted):")
    header = "".join(f"{n[:10]:>12}" for n in label_names)
    logger.info(f"{'':<24}{header}")
    for i, name in enumerate(label_names):
        row = "".join(f"{cm[i, j]:>12}" for j in range(len(label_names)))
        logger.info(f"{name[:22]:<24}{row}")

    off_diag_pairs = []
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            if i != j and cm[i, j] > 0:
                off_diag_pairs.append({'true': label_names[i], 'pred': label_names[j], 'count': int(cm[i, j])})
    off_diag_pairs.sort(key=lambda x: -x['count'])
    logger.info("\nTop confused pairs (true -> predicted):")
    for p in off_diag_pairs[:10]:
        logger.info(f"  {p['true']} -> {p['pred']}: {p['count']}")

    # --- Physics subtype (9-class, within "Reg signal RF hors tolerance") ---
    subtype_result = None
    subtype = pd.Series(data['y_physics_subtype'])
    has_subtype = subtype.notna()
    if has_subtype.sum() > 50:
        sub_names = sorted(subtype[has_subtype].unique())
        sub_map = {n: i for i, n in enumerate(sub_names)}
        y_sub = subtype[has_subtype].map(sub_map).values
        sub_data = {
            'features_all': data['features_all'].loc[has_subtype.values].reset_index(drop=True),
            'feature_cols': data['feature_cols'],
        }
        split_sub = leakage_safe_split(sub_data, y_sub, test_size=0.3, random_state=42, stratify=False)
        clf_sub = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_sub.fit(split_sub['X_train'], split_sub['y_train'])
        y_pred_sub = clf_sub.predict(split_sub['X_test'])
        cm_sub = confusion_matrix(split_sub['y_test'], y_pred_sub, labels=list(range(len(sub_names))))
        macro_f1_sub = float(f1_score(split_sub['y_test'], y_pred_sub, average='macro', zero_division=0))

        logger.info(f"\nPhysics-subtype (within 'Reg signal RF hors tolerance', n={int(has_subtype.sum())}) "
                    f"confusion matrix (rows=true, cols=predicted):")
        header = "".join(f"{n[:10]:>12}" for n in sub_names)
        logger.info(f"{'':<28}{header}")
        for i, name in enumerate(sub_names):
            row = "".join(f"{cm_sub[i, j]:>12}" for j in range(len(sub_names)))
            logger.info(f"{name[:26]:<28}{row}")
        logger.info(f"\nSubtype macro-F1 (single split): {macro_f1_sub:.3f}")

        subtype_result = {
            'n_events': int(has_subtype.sum()),
            'n_subtypes': len(sub_names),
            'subtype_names': sub_names,
            'macro_f1': macro_f1_sub,
            'confusion_matrix': cm_sub.tolist(),
        }

    return {
        'root_cause': {
            'confusion_matrix': cm.tolist(),
            'label_names': label_names,
            'top_confused_pairs': off_diag_pairs[:10],
        },
        'physics_subtype': subtype_result,
    }


def learning_curves():
    logger.info("=" * 78)
    logger.info("3. LEARNING CURVES")
    logger.info("=" * 78)

    data = load_v6()
    results = {}

    # Binary (step 06)
    y_binary = data['y_binary']
    X_raw, _ = raw_feature_matrix(data)
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X_raw)  # single global fit ok here: learning_curve
                                                        # re-splits internally each fold anyway and
                                                        # this is purely diagnostic, not a reported metric
    train_sizes = np.linspace(0.1, 1.0, 8)
    for label, y, name in [('binary', y_binary, 'Step 06 binary (RF)'),
                            ('root_cause', None, 'Step 08 root-cause (RF)')]:
        if label == 'root_cause':
            has_fault = data['y_multilabel'].sum(axis=1) > 0
            y_use = np.argmax(data['y_multilabel'], axis=1)[has_fault]
            X_use = X_scaled[has_fault]
        else:
            y_use = y
            X_use = X_scaled

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        sizes_abs, train_scores, test_scores = learning_curve(
            clf, X_use, y_use, train_sizes=train_sizes, cv=5, scoring='f1_macro',
            n_jobs=-1, random_state=42,
        )
        train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)

        logger.info(f"\n{name} learning curve (5-fold CV, f1_macro):")
        logger.info(f"{'n_train':>10} {'train_f1':>12} {'cv_f1':>12} {'gap':>8}")
        for n, trm, tem in zip(sizes_abs, train_mean, test_mean):
            logger.info(f"{int(n):>10} {trm:>12.3f} {tem:>12.3f} {trm - tem:>8.3f}")

        final_gap = float(train_mean[-1] - test_mean[-1])
        cv_slope = float(test_mean[-1] - test_mean[0])
        results[label] = {
            'train_sizes': [int(n) for n in sizes_abs],
            'train_f1_mean': [float(v) for v in train_mean],
            'cv_f1_mean': [float(v) for v in test_mean],
            'cv_f1_std': [float(v) for v in test_std],
            'final_train_cv_gap': final_gap,
            'cv_f1_slope_over_range': cv_slope,
        }
        logger.info(f"  Final train-CV gap: {final_gap:.3f}  |  CV F1 improvement 10%->100% of data: {cv_slope:.3f}")

    return results


def dimensionality_check():
    logger.info("=" * 78)
    logger.info("4. DIMENSIONALITY CHECK")
    logger.info("=" * 78)
    data = load_v6()
    n_features = len(data['feature_cols'])
    n_events = data['features_all'].shape[0]
    n_train = int(round(n_events * 0.7))
    ratio = n_features / n_train
    logger.info(f"Features: {n_features}, total events: {n_events}, ~train rows (70%): {n_train}")
    logger.info(f"Feature-to-train-row ratio: {ratio:.3f} (p/n; >0.1 is often considered a high-dimensional regime)")
    return {'n_features': n_features, 'n_events': n_events, 'n_train_approx': n_train,
            'feature_to_train_row_ratio': ratio}


def main():
    imbalance = imbalance_correlation()
    confusion = confusion_analysis()
    curves = learning_curves()
    dim = dimensionality_check()

    save_manifest(
        phase='10_phase3_weakness_diagnosis',
        metrics={
            'imbalance_pearson_r': {'value': imbalance['pearson_r_size_vs_f1'], 'fmt': '.3f',
                                     'label': 'Pearson r: category size vs mean F1'},
            'imbalance_pearson_r_log': {'value': imbalance['pearson_r_log_size_vs_f1'], 'fmt': '.3f',
                                         'label': 'Pearson r: log10(category size) vs mean F1'},
            'feature_to_train_row_ratio': {'value': dim['feature_to_train_row_ratio'], 'fmt': '.3f',
                                            'label': 'Feature-to-train-row ratio (p/n)'},
            'binary_learning_curve_final_gap': {'value': curves['binary']['final_train_cv_gap'], 'fmt': '.3f',
                                                 'label': 'Step 06 binary: final train-CV F1 gap'},
            'root_cause_learning_curve_final_gap': {'value': curves['root_cause']['final_train_cv_gap'], 'fmt': '.3f',
                                                      'label': 'Step 08 root-cause: final train-CV F1 gap'},
            'root_cause_learning_curve_cv_slope': {'value': curves['root_cause']['cv_f1_slope_over_range'], 'fmt': '.3f',
                                                    'label': 'Step 08 root-cause: CV F1 gain from 10% to 100% of data'},
        },
        pipeline_run={
            'dataset_version': 'V6',
            'dataset_path': str(V6_DIR / 'features_engineered_v6.pkl'),
            'script': 'pipeline/00_scripts/phase3_weakness_diagnosis.py',
        },
        meta={
            'imbalance_per_category': imbalance['per_category'],
            'confusion': confusion,
            'learning_curves': curves,
            'dimensionality': dim,
        },
    )
    logger.info("\nSaved manifest: analysis/results_manifest/10_phase3_weakness_diagnosis.yaml")


if __name__ == '__main__':
    main()
