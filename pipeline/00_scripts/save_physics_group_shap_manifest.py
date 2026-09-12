#!/usr/bin/env python3
"""
Wrap prepare_10_enhanced_shap_analysis.py's physics-group SHAP importance
output into a manifest.

V6 FIX (2026-09-07, two rounds of independent review):

Round 1 found the original grouping (PHYSICS_GROUPS: Amplitude/Phase/Power/
Frequency/Temporal/Statistical/Other, first-match substring heuristic) had
two problems: (1) it misclassified known features -- e.g. Ucav_mean/Ucav_std
(whole-signal Statistical features) into "Amplitude" because 'Ucav' matched
before any Statistical-suffix pattern was checked, and would have classified
a metadata field like meta_AMPT into "Amplitude" too; (2) the reported
group-SUM percentages structurally favor whichever group has more member
features (the 7 physics-informed families combined are 38 features;
Statistical is 256, Temporal is 466 -- a ~19:1 imbalance never controlled
for). prepare_10_enhanced_shap_analysis.py now classifies features by the
actual code section that produced them (verified 100% coverage, no "Other"
fallback) and reports the group SUM alongside the per-feature MEAN (dividing
out the count imbalance).

Round 2 found two further problems with round 1's own fix: (a) the "sum by
group" input was RandomForestClassifier.feature_importances_ (Gini/MDI
impurity importance), not real SHAP values, despite being labeled SHAP
throughout -- prepare_10_enhanced_shap_analysis.py now aggregates
mean(|shap_value|) per feature from the genuine shap.TreeExplainer output
instead; (b) the per-feature MEAN, in the small 4-8-feature physics groups,
can itself be dominated by a single outlier feature (as high as ~95% of a
group's own sum from one feature in some categories) -- this manifest now
also exposes the per-feature MEDIAN (far more robust to that failure mode)
so the mean is not presented as if it were bias-free.
"""
import os
import sys
from pathlib import Path
import json

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utilities.reporting.manifest import save_manifest

# 2026-09-10: was hardcoded to cooked_data_v6 -- parameterized the same way as
# the SHAP extraction scripts it reads output from (SPIRAL2_COOKED_DIR,
# default V6 for back-compat).
COOKED = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6'))
OUTPUT_DIR = COOKED / 'step_10_enhanced_shap'


def main():
    df_sum = pd.read_csv(OUTPUT_DIR / 'physics_group_importance.csv', index_col=0)
    df_mean = pd.read_csv(OUTPUT_DIR / 'physics_group_importance_per_feature_mean.csv', index_col=0)
    df_median = pd.read_csv(OUTPUT_DIR / 'physics_group_importance_per_feature_median.csv', index_col=0)
    counts = pd.read_csv(OUTPUT_DIR / 'physics_group_feature_counts.csv', index_col=0)['n_features']
    with open(OUTPUT_DIR / 'analysis_summary.json') as f:
        summary = json.load(f)

    print("Physics-group SHAP importance, per-feature-MEAN (%), per fault category:")
    print(df_mean.round(1).to_string())
    print("\nPhysics-group SHAP importance, per-feature-MEDIAN (%), per fault category:")
    print(df_median.round(1).to_string())
    print("\nPhysics-group SHAP importance, group SUM (%), per fault category:")
    print(df_sum.round(1).to_string())
    print("\nFeature counts per group:")
    print(counts.to_string())

    metrics = {}
    for category, row in df_mean.iterrows():
        slug = category.lower().replace(' ', '_').replace('/', '_')
        top_group = row.idxmax()
        metrics[f'{slug}_top_group'] = {'value': str(top_group), 'fmt': None,
                                          'label': f'{category}: dominant physics-group, per-feature-mean SHAP (%)'}
        metrics[f'{slug}_top_group_mean_pct'] = {'value': float(row[top_group]), 'fmt': '.1f',
                                                   'label': f'{category}: {top_group} share of per-feature-mean SHAP importance (%)'}
        # Also expose the sum-based top group for direct comparison, since it can differ from the mean-based one.
        sum_row = df_sum.loc[category]
        sum_top_group = sum_row.idxmax()
        metrics[f'{slug}_top_group_by_sum'] = {'value': str(sum_top_group), 'fmt': None,
                                                 'label': f'{category}: dominant physics-group, group-sum SHAP (%) -- may differ from the per-feature-mean top group'}
        metrics[f'{slug}_top_group_sum_pct'] = {'value': float(sum_row[sum_top_group]), 'fmt': '.1f',
                                                  'label': f'{category}: {sum_top_group} share of group-sum SHAP importance (%)'}
        # Median top group, exposed for direct comparison -- far more robust
        # to a single outlier feature than the mean, but degenerate (often
        # 0% or 100%) at these small (4-8 feature) group sizes, so reported
        # as a complementary check, not a replacement primary metric.
        median_row = df_median.loc[category]
        median_top_group = median_row.idxmax()
        metrics[f'{slug}_top_group_by_median'] = {'value': str(median_top_group), 'fmt': None,
                                                     'label': f'{category}: dominant physics-group, per-feature-median SHAP (%) -- robust to single-feature outliers but degenerate at small group sizes'}
        metrics[f'{slug}_top_group_median_pct'] = {'value': float(median_row[median_top_group]), 'fmt': '.1f',
                                                      'label': f'{category}: {median_top_group} share of per-feature-median SHAP importance (%)'}

    for group in df_mean.columns:
        metrics[f'n_features_{group.lower().replace(" ", "_").replace("/", "_")}'] = {
            'value': int(counts[group]), 'fmt': None,
            'label': f'Number of features classified into the {group} group',
        }
    metrics['n_features_physics_informed_total'] = {
        'value': int(counts[['Effective Decay', 'Detuning', 'Microphonics', 'RF Mismatch',
                              'Modulator Command', 'Phase Stability', 'RF Power Flow']].sum()),
        'fmt': None,
        'label': 'Total features across the seven named physics-informed families combined',
    }

    save_manifest(
        phase='10_physics_group_shap',
        metrics=metrics,
        pipeline_run={
            'dataset_version': (lambda _n: _n.upper() if _n else 'V2')(COOKED.name.replace('cooked_data', '').lstrip('_')),
            'dataset_path': str(COOKED / 'features_engineered.pkl'),
            'script': 'pipeline/00_scripts/prepare_10_enhanced_shap_analysis.py',
        },
        meta={
            'grouping_method': (
                "Exact-name / prefix / suffix classification by the actual "
                "engineer_features() code section that produced each feature "
                "(effective decay, detuning, microphonics, RF mismatch, "
                "modulator command, phase stability, RF power flow -- the "
                "seven families in Appendix A -- plus Statistical, Temporal, "
                "Metadata/Status, Other), verified 100% coverage with no "
                "'Other' fallback against the real 770-feature V6 set. "
                "Replaces a first-match substring heuristic that misclassified "
                "known features (see get_physics_group() in "
                "prepare_10_enhanced_shap_analysis.py)."
            ),
            'primary_metric_is_per_feature_mean': (
                "The seven physics-informed families total only "
                f"{int(counts[['Effective Decay','Detuning','Microphonics','RF Mismatch','Modulator Command','Phase Stability','RF Power Flow']].sum())} "
                f"features combined, versus {int(counts['Statistical'])} Statistical and {int(counts['Temporal'])} Temporal -- "
                "a group-SUM comparison structurally favors the larger groups "
                "regardless of per-feature informativeness, so the per-feature "
                "MEAN (dividing out this count imbalance) is the metric this "
                "manifest treats as primary; the sum-based table is exposed "
                "alongside it for direct comparison, not hidden."
            ),
            'full_mean_table_pct': df_mean.round(2).to_dict(orient='index'),
            'full_sum_table_pct': df_sum.round(2).to_dict(orient='index'),
            'feature_counts_per_group': counts.to_dict(),
            'per_label_test_scores': summary['test_scores'],
            'per_label_n_positive': summary['n_positive_samples'],
        },
    )
    print("\nSaved manifest: analysis/results_manifest/10_physics_group_shap.yaml")


if __name__ == '__main__':
    main()
