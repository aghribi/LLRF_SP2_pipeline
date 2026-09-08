#!/usr/bin/env python3
"""
Investigate step 05's (precursor detection) near-perfect supervised AUC.

Background: RF/LSTM precursor detectors score 0.99-1.0 AUC while genuinely
unsupervised methods (Isolation Forest/LOF/Mahalanobis/PCA-reconstruction)
score only 0.64-0.69 on the full feature set. This module's first version
hypothesized a pre-trigger-window-length confound: engineer_features()'s
"steady-state" pre/post features (detuning, RF-mismatch, modulator, RF-power)
deliberately use ALL available pre-trigger data (data[:zero_idx], unbounded),
and this script confirmed zero_idx's position in the buffer differs
systematically between Normal (~0.72 mean pre-trigger fraction) and Fault
(~0.49) events -- a real, replicated difference.

HOWEVER: engineer_features() also has a SEPARATE, already-correctly-engineered
"true precursor" feature group (section 4: trend/CUSUM/early-late-window
features, and section 6: preprocess_signals()'s 5 segments) that uses a FIXED
physical-duration window immediately before zero_idx (scale_samples(3000,
dt_us)) and explicitly `continue`s (skips) computing features for any event
whose available pre-trigger data is shorter than that window -- i.e. window
length is already controlled for in this feature group by design. Restricting
to ONLY this 382-column subset (out of 766) and re-running the supervised RF
still gives AUC = 0.9998 +/- 0.0002 (10 repeated splits), while Isolation
Forest/LOF on the SAME 382 columns still only reach 0.65/0.59 -- unchanged
from the full feature set. A window-length confound cannot explain this,
since window length is already fixed here.

Conclusion: the strong supervised/weak-unsupervised gap is not primarily an
artifact of the window-length difference measured below (which is real, but
affects the *unbounded* "steady-state" features, not the *bounded* "true
precursor" ones) -- it is the ordinary signature of a real, structured
decision boundary that supervised learning finds and density-based novelty
detection does not. The one remaining open question is INTERPRETIVE, not
statistical: this fixed window sits immediately before zero_idx (the
*informatic* ALM trigger), which per this codebase's own AMPT-refinement work
can lag the true physical fault onset -- so this likely reflects detection of
the fault already ramping up ahead of the interlock trip, not a strictly
independent precursor from before any physical change begins. See the
"corrected_conclusion" meta field below.

This script re-reads a stratified sample of RAW postmortem files (bypassing
the cached features_engineered_v6.pkl, which does not store zero_idx) and
recomputes zero_idx exactly as prepare_data_cluster_v6.py does
(zero_idx = argmin(abs(time_arr))), to measure the buffer-fraction gap.
"""
import sys
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/pbs/throng/m4cast/projects/SPIRAL2/programmes/PyPostMortem/src')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utilities.reporting.manifest import save_manifest

sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import leakage_safe_split, true_precursor_columns

V6_DIR = '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data_v6'
N_PER_CLASS = 40


def controlled_feature_scope_test(data, y):
    """
    Does restricting to ONLY the fixed-window "true precursor" features change
    the supervised-vs-unsupervised AUC gap? If window length were the whole
    story, fixing it here should collapse supervised AUC toward the
    unsupervised baseline. It does not (see module docstring).
    """
    cols = true_precursor_columns(data['feature_cols'])
    d = {'features_all': data['features_all'], 'feature_cols': cols}

    rf_aucs = []
    for seed in range(10):
        split = leakage_safe_split(d, y, test_size=0.3, random_state=seed, stratify=True)
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(split['X_train'], split['y_train'])
        proba = clf.predict_proba(split['X_test'])[:, 1]
        rf_aucs.append(roc_auc_score(split['y_test'], proba))
    rf_aucs = np.array(rf_aucs)

    iso_aucs, lof_aucs = [], []
    for seed in range(3):
        split = leakage_safe_split(d, y, test_size=0.3, random_state=seed, stratify=True)
        iso = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
        iso.fit(split['X_train'])
        iso_aucs.append(roc_auc_score(split['y_test'], -iso.score_samples(split['X_test'])))
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        lof.fit(split['X_train'])
        lof_aucs.append(roc_auc_score(split['y_test'], -lof.score_samples(split['X_test'])))

    result = {
        'n_columns': len(cols),
        'n_columns_total': len(data['feature_cols']),
        'rf_auc_mean': float(rf_aucs.mean()), 'rf_auc_std': float(rf_aucs.std()),
        'isolation_forest_auc_mean': float(np.mean(iso_aucs)),
        'lof_auc_mean': float(np.mean(lof_aucs)),
    }
    print(f"\n'True precursor' feature subset: {len(cols)} of {len(data['feature_cols'])} columns")
    print(f"  Supervised RF AUC:     mean={result['rf_auc_mean']:.4f} std={result['rf_auc_std']:.4f} (10 splits)")
    print(f"  Isolation Forest AUC:  mean={result['isolation_forest_auc_mean']:.4f} (3 splits)")
    print(f"  LOF AUC:               mean={result['lof_auc_mean']:.4f} (3 splits)")
    return result


def get_zero_idx(path):
    from PyPostmortem.utils.PyPostMortem import Read_Signals
    with open(path, 'rb') as f:
        content = f.read()
    _, time_arr, _, _, _, _ = Read_Signals(
        content, compute_defauts=False, compute_etats=False, plot_signaux=False, show_header=False
    )
    time_arr = np.asarray(time_arr)
    zero_idx = int(np.argmin(np.abs(time_arr)))
    return zero_idx, len(time_arr)


def main():
    with open(f'{V6_DIR}/features_engineered_v6.pkl', 'rb') as f:
        data = pickle.load(f)
    with open(f'{V6_DIR}/processed_file_paths.pkl', 'rb') as f:
        paths = pickle.load(f)

    y = data['y_binary']
    assert len(paths) == len(y), "processed_file_paths.pkl and y_binary must be same length/order"

    rng = np.random.RandomState(0)
    normal_idx = np.where(y == 0)[0]
    fault_idx = np.where(y == 1)[0]
    sample_normal = rng.choice(normal_idx, size=min(N_PER_CLASS, len(normal_idx)), replace=False)
    sample_fault = rng.choice(fault_idx, size=min(N_PER_CLASS, len(fault_idx)), replace=False)

    results = {}
    for label, sample in [('normal', sample_normal), ('fault', sample_fault)]:
        zis, lens, failed = [], [], 0
        for i in sample:
            try:
                zi, n = get_zero_idx(paths[i])
                zis.append(zi)
                lens.append(n)
            except Exception:
                failed += 1
        zis, lens = np.array(zis), np.array(lens)
        frac = zis / lens
        results[label] = {
            'n_files': len(zis), 'n_failed': failed,
            'zero_idx_mean': float(zis.mean()), 'zero_idx_median': float(np.median(zis)),
            'zero_idx_std': float(zis.std()),
            'pretrigger_fraction_mean': float(frac.mean()),
            'pretrigger_fraction_median': float(np.median(frac)),
            'pretrigger_fraction_std': float(frac.std()),
        }
        print(f"{label}: n={len(zis)} (failed={failed})")
        print(f"  zero_idx: mean={zis.mean():.0f} median={np.median(zis):.0f} std={zis.std():.0f}")
        print(f"  pre-trigger fraction (zero_idx/total_len): mean={frac.mean():.3f} median={np.median(frac):.3f}")

    diff = results['normal']['pretrigger_fraction_mean'] - results['fault']['pretrigger_fraction_mean']
    print(f"\nPre-trigger fraction gap (Normal - Fault): {diff:.3f}")

    scope_test = controlled_feature_scope_test(data, y)

    save_manifest(
        phase='05_precursor_window_leak_investigation',
        metrics={
            'normal_pretrigger_fraction_mean': {'value': results['normal']['pretrigger_fraction_mean'], 'fmt': '.3f',
                                                 'label': 'Normal events: mean pre-trigger fraction of buffer'},
            'fault_pretrigger_fraction_mean': {'value': results['fault']['pretrigger_fraction_mean'], 'fmt': '.3f',
                                                'label': 'Fault events: mean pre-trigger fraction of buffer'},
            'pretrigger_fraction_gap': {'value': diff, 'fmt': '.3f',
                                         'label': 'Gap in mean pre-trigger fraction (Normal minus Fault)'},
            'true_precursor_rf_auc': {'value': scope_test['rf_auc_mean'], 'fmt': '.4f',
                                       'label': 'Supervised RF AUC on the 382-col fixed-window precursor subset'},
            'true_precursor_isolation_forest_auc': {'value': scope_test['isolation_forest_auc_mean'], 'fmt': '.4f',
                                                     'label': 'Isolation Forest AUC on the same 382-col subset'},
            'true_precursor_lof_auc': {'value': scope_test['lof_auc_mean'], 'fmt': '.4f',
                                        'label': 'LOF AUC on the same 382-col subset'},
        },
        pipeline_run={
            'dataset_version': 'V6',
            'dataset_path': f'{V6_DIR}/features_engineered_v6.pkl',
            'script': 'pipeline/00_scripts/investigate_precursor_window_leak.py',
        },
        meta={
            'n_per_class_sampled': N_PER_CLASS,
            'method': (
                'Re-read raw postmortem files directly (bypassing the cached feature pkl), '
                're-ran zero_idx = argmin(abs(time_arr)) exactly as prepare_data_cluster_v6.py, '
                'and computed pre-trigger fraction = zero_idx / total_samples per file.'
            ),
            'conclusion': (
                'Normal events have a substantially larger pre-trigger fraction than Fault events '
                '(gap ~0.21-0.25), confirmed via two independent raw-file samples. This is real and '
                'affects engineer_features()\'s "steady-state" pre/post features (detuning, RF-mismatch, '
                'modulator, RF-power), which deliberately use ALL available pre-trigger data with no '
                'fixed bound.'
            ),
            'corrected_conclusion': (
                'This window-length gap does NOT explain step 05\'s near-perfect supervised AUC, however: '
                'restricting to ONLY the codebase\'s already-correctly-bounded "true precursor" features '
                '(engineer_features() section 4 trend/CUSUM/early-late-window + section 6 segment-based, '
                '382 of 766 columns, all built on a FIXED scale_samples(3000, dt_us) window with a '
                '`continue`-if-too-short guard against variable window length) still gives supervised RF '
                'AUC=0.9998+/-0.0002 across 10 repeated splits, while Isolation Forest/LOF on the SAME '
                '382 columns still only reach 0.65/0.59 -- identical to their score on the full 766-column '
                'set. Window length cannot explain a result on features where window length is already '
                'fixed. The strong-supervised/weak-unsupervised gap is the ordinary signature of a real, '
                'learnable decision boundary, not an artifact. The remaining caveat is interpretive: this '
                'fixed window sits immediately before zero_idx (the informatic ALM trigger), which this '
                "codebase's own AMPT-refinement work already establishes can lag the true physical fault "
                'onset for at least one category -- so this likely reflects detecting the fault already '
                'ramping up ahead of the interlock trip, not an independent precursor from before any '
                'physical change begins. Recommendation: step 05/05a should be restricted to this 382-'
                'column "true precursor" subset (excluding whole-signal and unbounded steady-state '
                'features, which ARE inappropriate for this task) and the result reported as early '
                'detection of fault onset ahead of the interlock, not as prediction before any physical '
                'change -- a real, valuable, but differently-scoped claim than "precursor detection" '
                'currently implies.'
            ),
            'full_result': results,
            'feature_scope_test': scope_test,
        },
    )
    print("\nSaved manifest: analysis/results_manifest/05_precursor_window_leak_investigation.yaml")


if __name__ == '__main__':
    main()
