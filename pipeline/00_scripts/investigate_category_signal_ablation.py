#!/usr/bin/env python3
"""
Category-vs-Normal early detection, blind to the category's own defining signal(s).

For a given fault category, trains the same pre-trigger-only binary classifier
comparison as prepare_06b_phase1_binary_pretrigger.py (LogReg/RF/XGBoost/SVM), but
restricted to just that category's events vs. Normal, and with the raw signal(s) the
deterministic ground-truth classifier's OWN detection criterion for that category
reads (per Classify_PostMortemFile/src/classification.py) removed from the feature
set -- not just the literal column, every feature derived from that channel.

Purpose: distinguish a genuine independent early-warning signal (still detectable
without the category's own defining quantity) from the model just implicitly reading
out the same physical quantity that will trip the deterministic cut.

Category -> defining raw channel(s), verified directly in classification.py:
  Claquage ou quench cavité        -> Ucav
  Dép seuil de sécurité RF          -> Ucav, A Ucr   (UCAT is a per-file scalar header
                                       value, not an engineered feature -- confirmed
                                       absent from feature_cols, nothing to exclude)
  Seuil de vide                     -> vide, courant pickup
  Rég signal RF hors tolérance      -> Uci, A Ucr, Ucav, IQ  (Presence faisceau/beam
                                       presence is not an engineered feature either)
  Seuil pick-up                     -> courant pickup   (per project lead, 2026-09-09)
  Coupure externe rapide            -> no single defining channel in code; run as a
                                       full per-channel sweep instead (--exclude-channels
                                       one channel at a time across all 27)
  Absence autorisation RF           -> same, full per-channel sweep

Usage:
    python investigate_category_signal_ablation.py \
        --input /path/to/cooked_data_v7 \
        --category "Claquage ou quench cavité" \
        --exclude-channels "Ucav" \
        --tag quench_excl_Ucav \
        --output /path/to/cooked_data_v7/step_signal_ablation
"""

import sys
import argparse
from pathlib import Path
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utilities.reporting.manifest import save_manifest

sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import leakage_safe_split, true_precursor_columns

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--category', required=True, help='exact fault_column_names entry')
    parser.add_argument('--exclude-channels', default='',
                         help='comma-separated raw channel names to exclude, empty = baseline (no exclusion)')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    with open(input_dir / 'features_engineered.pkl', 'rb') as f:
        data = pickle.load(f)

    fault_names = list(data['fault_column_names'])
    if args.category not in fault_names:
        raise ValueError(f"Unknown category {args.category!r}; known: {fault_names}")
    cat_idx = fault_names.index(args.category)
    y_multilabel = data['y_multilabel']
    y_binary = data['y_binary']

    # This category's events vs. Normal only (not vs. every other fault) -- same scope
    # as step 06/06b's Fault-vs-Normal design, just restricted to one category.
    mask = (y_multilabel[:, cat_idx] == 1) | (y_binary == 0)
    y_sub = (y_multilabel[mask, cat_idx] == 1).astype(int)
    n_cat = int(y_sub.sum())
    n_normal = int((y_sub == 0).sum())
    print(f"Category '{args.category}': {n_cat} events vs {n_normal} Normal", flush=True)

    all_cols = data['feature_cols']
    pretrigger_cols = true_precursor_columns(all_cols)

    exclude_channels = [c.strip() for c in args.exclude_channels.split(',') if c.strip()]
    if exclude_channels:
        final_cols = [c for c in pretrigger_cols
                      if not any(c == ch or c.startswith(ch + '_') for ch in exclude_channels)]
    else:
        final_cols = pretrigger_cols
    n_excluded = len(pretrigger_cols) - len(final_cols)
    print(f"Pre-trigger columns: {len(pretrigger_cols)}, excluding channels {exclude_channels}: "
          f"removed {n_excluded}, using {len(final_cols)}", flush=True)

    features_all_sub = data['features_all'].iloc[mask].reset_index(drop=True)
    pkl_sub = {'features_all': features_all_sub, 'feature_cols': final_cols}

    metrics = {
        'n_category_events': {'value': n_cat, 'fmt': None, 'label': 'Category events'},
        'n_normal_events': {'value': n_normal, 'fmt': None, 'label': 'Normal events (comparison set)'},
        'n_features_used': {'value': len(final_cols), 'fmt': None, 'label': 'Pre-trigger features used'},
        'n_features_excluded': {'value': n_excluded, 'fmt': None, 'label': 'Features excluded (channel-derived)'},
    }

    min_class_count = min(n_cat, n_normal)
    if min_class_count < 10:
        print(f"SKIP modeling: smallest class has only {min_class_count} events, "
              f"too few for a meaningful train/test split.", flush=True)
    else:
        split = leakage_safe_split(pkl_sub, y_sub, test_size=0.3, random_state=42, stratify=True)
        X_train, X_test = split['X_train'], split['X_test']
        y_train, y_test = split['y_train'], split['y_test']

        models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        }
        if XGBOOST_AVAILABLE:
            models['xgboost'] = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                               random_state=42, n_jobs=-1)
        models['svm'] = SVC(kernel='rbf', probability=True, random_state=42)

        for name, model in models.items():
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                print(f"  SKIP {name}: not enough class diversity in this split", flush=True)
                continue
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            print(f"  {name}: acc={acc:.4f} auc={auc:.4f} f1={f1:.4f}", flush=True)
            metrics[f'{name}_accuracy'] = {'value': float(acc), 'fmt': '.1%', 'label': f'{name} accuracy'}
            metrics[f'{name}_roc_auc'] = {'value': float(auc), 'fmt': '.4f', 'label': f'{name} ROC AUC'}
            metrics[f'{name}_f1'] = {'value': float(f1), 'fmt': '.3f', 'label': f'{name} F1'}

    dataset_version = (lambda _n: _n.upper() if _n else 'V2')(
        input_dir.name.replace('cooked_data', '').lstrip('_'))

    save_manifest(
        phase=f'signal_ablation_{args.tag}',
        metrics=metrics,
        pipeline_run={
            'dataset_version': dataset_version,
            'dataset_path': str(input_dir / 'features_engineered.pkl'),
            'script': 'pipeline/00_scripts/investigate_category_signal_ablation.py',
        },
        meta={
            'category': args.category,
            'excluded_channels': exclude_channels,
            'purpose': (
                "Category-vs-Normal, pre-trigger-only, testing whether early detection "
                "survives excluding the raw signal(s) this category's own deterministic "
                "classification criterion is based on -- distinguishes a real independent "
                "early-warning signal from the model just reading out the same quantity "
                "that will trip the deterministic cut."
            ),
        },
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f'{args.tag}.pkl', 'wb') as f:
        pickle.dump({'metrics': metrics, 'excluded_channels': exclude_channels}, f)

    print(f"\nSTEP COMPLETE: {args.tag}", flush=True)


if __name__ == '__main__':
    main()
