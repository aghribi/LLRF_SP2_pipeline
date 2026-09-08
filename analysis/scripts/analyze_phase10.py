#!/usr/bin/env python
"""
Phase 10 Comprehensive Pipeline Dashboard - Final Report
Integrates all pipeline phases into a unified analysis
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')
RESULTS_FILE = OUTPUT_DIR / 'step_10_comprehensive' / 'comprehensive_analysis.pkl'
FIGURES_DIR = Path('../outputs')
FIGURES_DIR.mkdir(exist_ok=True)

# Visualization setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("PHASE 10: COMPREHENSIVE PIPELINE DASHBOARD")
print("="*80)
print()

# Load results
print(f"Loading comprehensive results from: {RESULTS_FILE}")
with open(RESULTS_FILE, 'rb') as f:
    results = pickle.load(f)

summary = results['summary']
all_phases = results['all_phases']
report_text = results['report']

print(f"✓ Results loaded")
print(f"\nPipeline Version: {summary['pipeline_version']}")
print(f"Phases Completed: {len(summary['phases_completed'])}")
for phase in summary['phases_completed']:
    print(f"  ✓ {phase.upper()}")
print()

#========================================
# PHASE SUMMARY TABLE
#========================================
print("="*80)
print("PIPELINE SUMMARY - ALL PHASES")
print("="*80)
print()

phase_summary = []

# Anomaly Detection (Phase 03)
if all_phases['anomaly'] is not None:
    anomaly_data = all_phases['anomaly']
    n_anomalies_if = len(anomaly_data['isolation_forest']['predictions'])
    phase_summary.append({
        'Phase': '03 - Anomaly Detection',
        'Method': 'Isolation Forest',
        'Key Metric': f'{n_anomalies_if} events analyzed',
        'Status': '✓ Complete'
    })

# Precursor Detection (Phase 04-06)
if all_phases['precursor'] is not None:
    precursor_data = all_phases['precursor']
    y_test = precursor_data['data_splits']['y_test']
    y_pred = precursor_data['random_forest']['predictions']
    prec_acc = accuracy_score(y_test, y_pred)
    phase_summary.append({
        'Phase': '04-06 - Precursor Detection',
        'Method': 'Random Forest',
        'Key Metric': f'Accuracy: {prec_acc:.1%}',
        'Status': '✓ Complete'
    })

# Binary Classification (Phase 01-02)
if all_phases['binary'] is not None:
    binary_data = all_phases['binary']
    y_test = binary_data['data_splits']['y_test']
    y_pred_rf = binary_data['random_forest']['predictions']
    bin_acc = accuracy_score(y_test, y_pred_rf)
    phase_summary.append({
        'Phase': '01-02 - Binary Classification',
        'Method': 'Random Forest',
        'Key Metric': f'Accuracy: {bin_acc:.1%}',
        'Status': '✓ Complete'
    })

# Multi-label Classification (Phase 07)
if all_phases['multilabel'] is not None:
    multilabel_data = all_phases['multilabel']
    hamming = multilabel_data['hamming_loss']
    n_faults = len(multilabel_data['fault_names'])
    phase_summary.append({
        'Phase': '07 - Multi-label Classification',
        'Method': 'Multi-output RF',
        'Key Metric': f'Hamming Loss: {hamming:.3f} ({n_faults} faults)',
        'Status': '✓ Complete'
    })

# Root Cause Identification (Phase 08)
if all_phases['rootcause'] is not None:
    rootcause_data = all_phases['rootcause']
    y_test = rootcause_data['data_splits']['y_test']
    y_pred = rootcause_data['predictions']
    rc_acc = accuracy_score(y_test, y_pred)
    phase_summary.append({
        'Phase': '08 - Root Cause Identification',
        'Method': 'Random Forest',
        'Key Metric': f'Accuracy: {rc_acc:.1%}',
        'Status': '✓ Complete'
    })

# Fault Subtype Clustering (Phase 09)
if all_phases['clustering'] is not None:
    clustering_data = all_phases['clustering']
    n_fault_types = len(clustering_data)
    total_subtypes = sum(clustering_data[ft]['n_clusters'] for ft in clustering_data.keys())
    mean_silh = np.mean([clustering_data[ft]['silhouette'] for ft in clustering_data.keys()])
    phase_summary.append({
        'Phase': '09 - Fault Subtype Clustering',
        'Method': 'K-Means',
        'Key Metric': f'{total_subtypes} subtypes, Silh={mean_silh:.2f}',
        'Status': '✓ Complete'
    })

summary_df = pd.DataFrame(phase_summary)
print(summary_df.to_string(index=False))
print()

#========================================
# DETAILED METRICS BY PHASE
#========================================
print("="*80)
print("DETAILED PERFORMANCE METRICS")
print("="*80)
print()

# Binary Classification Details
if all_phases['binary'] is not None:
    print("--- BINARY CLASSIFICATION (Fault vs No-Fault) ---")
    binary_data = all_phases['binary']
    y_test = binary_data['data_splits']['y_test']

    methods = ['logistic_regression', 'random_forest', 'xgboost', 'svm']
    method_names = ['Logistic Reg', 'Random Forest', 'XGBoost', 'SVM']

    for method, name in zip(methods, method_names):
        if method in binary_data and binary_data[method] is not None:
            y_pred = binary_data[method]['predictions']
            acc = accuracy_score(y_test, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
            print(f"  {name:15s}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
    print()

# Multi-label Details
if all_phases['multilabel'] is not None:
    print("--- MULTI-LABEL CLASSIFICATION (Trigger Identification) ---")
    multilabel_data = all_phases['multilabel']
    fault_names = multilabel_data['fault_names']
    print(f"  Fault Types: {len(fault_names)}")
    print(f"  Hamming Loss: {multilabel_data['hamming_loss']:.3f}")
    print(f"  Test Events: {len(multilabel_data['data_splits']['y_test'])}")
    print()

# Root Cause Details
if all_phases['rootcause'] is not None:
    print("--- ROOT CAUSE IDENTIFICATION (Multi-fault Events) ---")
    rootcause_data = all_phases['rootcause']
    y_test = rootcause_data['data_splits']['y_test']
    y_pred = rootcause_data['predictions']
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print(f"  Test Events: {len(y_test)}")
    print()

# Clustering Details
if all_phases['clustering'] is not None:
    print("--- FAULT SUBTYPE CLUSTERING ---")
    clustering_data = all_phases['clustering']
    for fault_type in clustering_data.keys():
        fault_cluster = clustering_data[fault_type]
        n_clusters = fault_cluster['n_clusters']
        silhouette = fault_cluster['silhouette']
        n_events = len(fault_cluster['labels'])
        print(f"  {fault_type:30s}: {n_clusters} subtypes, Silh={silhouette:.2f}, Events={n_events}")
    print()

#========================================
# INTEGRATED PIPELINE FLOW VISUALIZATION
#========================================
print("="*80)
print("PIPELINE ARCHITECTURE")
print("="*80)
print()

# Create pipeline flow diagram (text-based)
pipeline_flow = """
SPIRAL2 LLRF ANOMALY DETECTION PIPELINE FLOW
============================================

                    ┌──────────────────────────┐
                    │  Raw LLRF Time Series    │
                    │  (Ucav, PhaseCav, etc.)  │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Feature Engineering     │
                    │  (Domain + Statistical)  │
                    └────────────┬─────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
    ┌───────────▼──────────┐ ┌──▼──────────┐ ┌──▼──────────────┐
    │ Phase 03: Anomaly    │ │ Phase 04-06 │ │ Phase 01-02     │
    │ Detection            │ │ Precursor   │ │ Binary          │
    │ (IF, LOF, DBSCAN)    │ │ Detection   │ │ Classification  │
    └───────────┬──────────┘ └──┬──────────┘ └──┬──────────────┘
                │                │                │
                └────────────────┼────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Phase 07: Multi-label   │
                    │  Trigger Identification  │
                    │  (Which faults active?)  │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Phase 08: Root Cause    │
                    │  Identification          │
                    │  (Primary fault?)        │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Phase 09: Subtype       │
                    │  Clustering              │
                    │  (Fault mechanisms)      │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Phase 10: Dashboard     │
                    │  Comprehensive Analysis  │
                    └──────────────────────────┘
"""

print(pipeline_flow)

#========================================
# KEY FINDINGS AND RECOMMENDATIONS
#========================================
print("\n" + "="*80)
print("KEY FINDINGS AND RECOMMENDATIONS")
print("="*80)
print()

print("**Performance Highlights**:")
if all_phases['binary'] is not None:
    print(f"  ✓ Binary classification: {bin_acc:.1%} accuracy (fault detection)")
if all_phases['multilabel'] is not None:
    print(f"  ✓ Multi-label: {multilabel_data['hamming_loss']:.3f} Hamming loss (7 fault types)")
if all_phases['rootcause'] is not None:
    print(f"  ✓ Root cause: {rc_acc:.1%} accuracy (multi-fault events)")
if all_phases['clustering'] is not None:
    print(f"  ✓ Clustering: {total_subtypes} fault subtypes discovered")
print()

print("**Production Deployment Strategy**:")
print("  1. Binary Classification → Real-time fault detection")
print("  2. Multi-label → Identify all active fault triggers")
print("  3. Root Cause → Determine primary cause in multi-fault scenarios")
print("  4. Clustering → Severity classification and detailed diagnostics")
print()

print("**Recommended Workflow**:")
print("  Step 1: Monitor incoming events with binary classifier")
print("  Step 2: For detected faults, run multi-label to identify active triggers")
print("  Step 3: If multiple triggers, apply root cause analysis")
print("  Step 4: Match to subtype cluster for severity/mechanism")
print("  Step 5: Alert operators with full diagnostic information")
print()

print("**Next Steps for Production**:")
print("  1. Temporal onset detection for improved root cause labeling")
print("  2. Physics-based rule validation")
print("  3. Confidence thresholds and uncertainty quantification")
print("  4. Online learning for model updates")
print("  5. Integration with EPICS control system")
print()

#========================================
# SAVE SUMMARY REPORT
#========================================
report_file = FIGURES_DIR / 'phase10_comprehensive_report.txt'

with open(report_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("SPIRAL2 LLRF ANOMALY DETECTION PIPELINE - COMPREHENSIVE REPORT\n")
    f.write("="*80 + "\n\n")

    f.write(f"Pipeline Version: {summary['pipeline_version']}\n")
    f.write(f"Phases Completed: {len(summary['phases_completed'])}\n\n")

    f.write("PHASE SUMMARY:\n")
    f.write(summary_df.to_string(index=False) + "\n\n")

    f.write("PIPELINE ARCHITECTURE:\n")
    f.write(pipeline_flow + "\n\n")

    f.write("KEY METRICS:\n")
    if all_phases['binary'] is not None:
        f.write(f"  Binary Classification: {bin_acc:.1%}\n")
    if all_phases['multilabel'] is not None:
        f.write(f"  Multi-label Hamming Loss: {multilabel_data['hamming_loss']:.3f}\n")
    if all_phases['rootcause'] is not None:
        f.write(f"  Root Cause Accuracy: {rc_acc:.1%}\n")
    if all_phases['clustering'] is not None:
        f.write(f"  Total Subtypes: {total_subtypes}, Mean Silhouette: {mean_silh:.2f}\n")

print(f"✓ Comprehensive report saved to: {report_file}")

#========================================
# FINAL SUMMARY
#========================================
print("\n" + "="*80)
print("PHASE 10 COMPLETE - PIPELINE DASHBOARD GENERATED")
print("="*80)
print(f"✓ All {len(summary['phases_completed'])} phases integrated")
print(f"✓ Comprehensive report saved")
print(f"✓ Ready for production deployment planning")
print("="*80)
