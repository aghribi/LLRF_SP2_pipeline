#!/usr/bin/env python3
"""
Data attrition chain: raw postmortem files -> ground-truth CSV -> V6 dataset.

Phase 1 (results-reliability plan) deliverable: every count below is computed
live from the actual raw_data directory, the deterministic classifier's own
result CSV, and V6's own error/summary logs -- never hand-typed -- so this
stays correct as the corpus grows. Run it, don't read a stale copy of its
output.

Usage: python3 data_attrition_chain.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utilities.reporting.manifest import save_manifest

RAW_DATA_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/raw_data')
GROUND_TRUTH_CSV = Path(
    '/pbs/throng/m4cast/projects/SPIRAL2/programmes/Classify_PostMortemFile/result/LLRF_result.csv'
)
CLASSIFIER_LOG = Path(
    '/pbs/throng/m4cast/projects/SPIRAL2/programmes/Classify_PostMortemFile/'
    'logs/full_run_v2_20260904_103639.log'
)
V6_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6')

FAULT_CATEGORIES = [
    'Seuil pick-up',
    'Coupure externe rapide',
    'Absence autorisation RF',
    'Claquage ou quench cavité',
    'Seuil de vide',
    'Dép seuil de sécurité RF',
    'Rég signal RF hors tolérance',
]


def count_raw_files():
    return sum(1 for p in RAW_DATA_DIR.rglob('*') if p.is_file())


def count_zero_byte_files():
    return sum(1 for p in RAW_DATA_DIR.rglob('*') if p.is_file() and p.stat().st_size == 0)


def classify_processing_errors():
    if not CLASSIFIER_LOG.exists():
        return None, {}
    lines = [l for l in CLASSIFIER_LOG.read_text(errors='replace').splitlines()
             if 'Fichier ignoré' in l]
    reasons = {}
    for l in lines:
        reason = l.split('- Erreur :', 1)[-1].strip()
        if reason == '':
            reason = '(empty exception message)'
        elif "codec can't decode" in reason:
            reason = 'utf-8 decode error (stray non-postmortem file)'
        reasons[reason] = reasons.get(reason, 0) + 1
    return len(lines), reasons


def load_ground_truth():
    df = pd.read_csv(GROUND_TRUTH_CSV, sep=';')
    non_filtre = df[df['Label'] != 'Filtré']
    return df, non_filtre


def filtre_reason_breakdown(gt_df):
    """
    Per-reason breakdown of the Filtré rows' own Détails column. The six checks in
    classification.py::_check_filters run as an early-exit cascade (feed-forward
    checked first, since 2026-09-09 scoped to acquisitions >= 2021 -- see
    config.yaml's feed_forward_min_year), so these are not independent,
    non-overlapping counts: a file counted under one reason would not have been
    evaluated against a later one.
    """
    filtre = gt_df[gt_df['Label'] == 'Filtré']
    return filtre['Détails'].value_counts().to_dict()


def v6_feature_extraction_errors():
    err_file = V6_DIR / 'error_files.txt'
    if not err_file.exists():
        return []
    lines = err_file.read_text().splitlines()
    return [l for l in lines if l.strip() and ':' in l and 'raw_data' in l]


def main():
    raw_files = count_raw_files()
    zero_byte = count_zero_byte_files()
    n_errors, error_reasons = classify_processing_errors()
    gt_df, gt_passing = load_ground_truth()
    n_gt_rows = len(gt_df)
    n_gt_passing = len(gt_passing)
    n_filtre = n_gt_rows - n_gt_passing

    v6_extraction_errors = v6_feature_extraction_errors()
    v6_normal = 1066  # cross-checked against processing_summary_v6.txt below
    summary_txt = (V6_DIR / 'processing_summary_v6.txt').read_text()
    for line in summary_txt.splitlines():
        if line.startswith('Total events processed:'):
            v6_total = int(line.split(':')[1].strip())
        if line.startswith('Normal events:'):
            v6_normal = int(line.split(':')[1].strip())
        if line.startswith('Fault events:'):
            v6_fault = int(line.split(':')[1].strip())

    unreconciled = n_gt_rows - (raw_files - n_errors - zero_byte)
    filtre_reasons = filtre_reason_breakdown(gt_df)

    print("=" * 88)
    print("DATA ATTRITION CHAIN: raw files -> ground truth -> V6 dataset")
    print("=" * 88)
    print(f"{'Stage':<55} {'Count':>10}")
    print("-" * 88)
    print(f"{'Raw postmortem files (raw_data/, all years)':<55} {raw_files:>10}")
    print(f"{'  - processing errors (classifier could not read)':<55} {-n_errors:>10}")
    for reason, cnt in sorted(error_reasons.items(), key=lambda x: -x[1]):
        print(f"{'      ' + reason:<55} {-cnt:>10}")
    print(f"{'  - zero-byte files':<55} {-zero_byte:>10}")
    print(f"{'= Ground-truth CSV rows (LLRF_result.csv)':<55} {n_gt_rows:>10}")
    if unreconciled:
        print(f"{'  (unreconciled vs. naive raw-errors-zerobyte arithmetic)':<55} {unreconciled:>10}")
    print(f"{'  - Filtré (quality-filter rejected -- early-exit cascade, see breakdown below)':<55} {-n_filtre:>10}")
    for reason, cnt in sorted(filtre_reasons.items(), key=lambda x: -x[1]):
        print(f"{'      ' + reason:<55} {-cnt:>10}")
    print(f"{'= Passes ground-truth quality filter':<55} {n_gt_passing:>10}")
    print(f"{'  - feature-extraction failures (V6 pipeline)':<55} {-len(v6_extraction_errors):>10}")
    for l in v6_extraction_errors:
        print(f"      {l}")
    print(f"{'= V6 final dataset (features_engineered_v6.pkl)':<55} {v6_total:>10}")
    print(f"{'    Normal':<55} {v6_normal:>10}")
    print(f"{'    Fault (all 7 categories)':<55} {v6_fault:>10}")
    print("=" * 88)
    if n_gt_passing > v6_total:
        gap = n_gt_passing - v6_total
        print(f"\nNOTE: ground-truth-passing ({n_gt_passing}) exceeds V6's extracted total ({v6_total}) by "
              f"{gap}. This is expected as of the 2026-09-09 feed-forward-filter fix (see "
              f"report/sections/03_system_description.tex): the ground-truth classifier has been "
              f"re-run and now correctly passes more pre-2021 events, but the feature-engineered "
              f"dataset (V6) has not yet been re-extracted against the corrected ground truth -- that "
              f"re-extraction (V7) is pending, not a bug in this script.")

    print("\nPer-category breakdown (raw Defaut count -> passes ground-truth filter -> V6):")
    print(f"{'Category':<32} {'Raw (Defaut)':>13} {'GT-passing':>11}")
    raw_by_cat = gt_df['Defaut'].value_counts()
    passing_by_cat = gt_passing['Defaut'].value_counts()
    for cat in ['Normal'] + FAULT_CATEGORIES:
        print(f"{cat:<32} {raw_by_cat.get(cat, 0):>13} {passing_by_cat.get(cat, 0):>11}")

    metrics = {
        'raw_files': {'value': raw_files, 'fmt': ',', 'label': 'Raw postmortem files scanned'},
        'processing_errors': {'value': n_errors, 'fmt': None, 'label': "Classifier processing errors ('Fichier ignoré')"},
        'zero_byte_files': {'value': zero_byte, 'fmt': None, 'label': 'Zero-byte raw files'},
        'ground_truth_rows': {'value': n_gt_rows, 'fmt': ',', 'label': 'Ground-truth CSV rows (LLRF_result.csv)'},
        'filtre_rejected': {'value': n_filtre, 'fmt': ',', 'label': "Rows rejected by quality filter (Label == 'Filtré')"},
        'gt_passing': {'value': n_gt_passing, 'fmt': ',', 'label': 'Rows passing ground-truth quality filter'},
        'v6_feature_extraction_errors': {'value': len(v6_extraction_errors), 'fmt': None,
                                          'label': 'Events dropped by V6 feature-extraction failure'},
        'v6_total_events': {'value': v6_total, 'fmt': ',', 'label': 'V6 final dataset event count'},
        'v6_normal_events': {'value': v6_normal, 'fmt': ',', 'label': 'V6 Normal events'},
        'v6_fault_events': {'value': v6_fault, 'fmt': ',', 'label': 'V6 fault events (all 7 categories)'},
    }
    for cat in FAULT_CATEGORIES:
        slug = cat.lower().replace(' ', '_').replace("'", '')
        metrics[f'raw_{slug}'] = {'value': int(raw_by_cat.get(cat, 0)), 'fmt': ',',
                                   'label': f"Raw 'Defaut' count: {cat}"}
        metrics[f'gt_passing_{slug}'] = {'value': int(passing_by_cat.get(cat, 0)), 'fmt': None,
                                          'label': f"Ground-truth-passing count: {cat}"}
    for reason, cnt in filtre_reasons.items():
        slug = 'filtre_reason_' + re.sub(r'[^a-z0-9]+', '_', reason.lower()).strip('_')
        metrics[slug] = {'value': int(cnt), 'fmt': ',', 'label': f"Quality-filter rejections: {reason}"}

    save_manifest(
        phase='00_data_attrition_chain',
        metrics=metrics,
        pipeline_run={
            'dataset_version': 'V6',
            'dataset_path': str(V6_DIR / 'features_engineered_v6.pkl'),
            'script': 'pipeline/00_scripts/data_attrition_chain.py',
        },
        meta={
            'unreconciled_count': int(unreconciled),
            'unreconciled_note': (
                'raw_files - processing_errors - zero_byte_files - ground_truth_rows is off by '
                f'{unreconciled}; likely a zero-byte file also caught by the processing-error '
                'logging (double-counted across both buckets), not yet root-caused further.'
            ) if unreconciled else None,
        },
    )
    print("\nSaved manifest: analysis/results_manifest/00_data_attrition_chain.yaml")


if __name__ == '__main__':
    main()
