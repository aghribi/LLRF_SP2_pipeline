#!/usr/bin/env python
"""
Execute multilabel_phases_comparison.ipynb cells - Comprehensive Multi-Label Cross-Phase Comparison
Compares all phases: A (BR+CC), B (CC optimized), C (Label Powerset), D (Deep Learning)
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (14, 6)

print("="*70)
print("MULTILABEL PHASES COMPARISON: ALL PHASES A, B, C, D")
print("="*70)
print("\n✓ Imports complete\n")

# Paths
COOKED = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')
OUT_DIR = Path('analysis/outputs/multilabel_comparison')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('Cooked data directory:', COOKED)
print('Output directory:', OUT_DIR)

# Load Phase 2 Results (Baseline: Multi-Output Random Forest)
print("\n" + "="*70)
print("LOADING PHASE 2 (BASELINE)")
print("="*70)

phase2_file = COOKED / 'step_07_phase2' / 'multilabel_triggers.pkl'
phase2 = None

if phase2_file.exists():
    with open(phase2_file, 'rb') as f:
        phase2_data = pickle.load(f)

    from sklearn.metrics import hamming_loss, f1_score

    y_test = phase2_data['data_splits']['y_test']
    y_pred = phase2_data['predictions']

    hamming = hamming_loss(y_test, y_pred)
    micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    phase2 = {
        'hamming_loss': hamming,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'y_test': y_test,
        'y_pred': y_pred,
        'fault_names': phase2_data.get('fault_names', [])
    }

    print("✓ Phase 2 (Multi-Output Random Forest Baseline) loaded")
    print(f"  Hamming Loss: {hamming:.6f}")
    print(f"  Micro F1: {micro_f1:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
else:
    print(f"✗ Phase 2 results not found at {phase2_file}")

# Load Phase A Results (BR + CC)
print("\n" + "="*70)
print("LOADING PHASE A (BR + CC)")
print("="*70)

phaseA_file = COOKED / 'step_07_phaseA_br' / 'phaseA_summary.pkl'
phaseA = None

if phaseA_file.exists():
    with open(phaseA_file, 'rb') as f:
        phaseA = pickle.load(f)

    print("✓ Phase A (Binary Relevance + Classifier Chain) loaded")
    if 'br' in phaseA:
        print(f"  BR - Hamming: {phaseA['br'].get('hamming_loss', 'N/A'):.6f}, "
              f"Micro F1: {phaseA['br'].get('micro_f1', 'N/A'):.4f}, "
              f"Macro F1: {phaseA['br'].get('macro_f1', 'N/A'):.4f}")
    if 'cc' in phaseA:
        print(f"  CC - Hamming: {phaseA['cc'].get('hamming_loss', 'N/A'):.6f}, "
              f"Micro F1: {phaseA['cc'].get('micro_f1', 'N/A'):.4f}, "
              f"Macro F1: {phaseA['cc'].get('macro_f1', 'N/A'):.4f}")
else:
    print(f"✗ Phase A results not found at {phaseA_file}")

# Load Phase B Results (Classifier Chains - Optimized)
print("\n" + "="*70)
print("LOADING PHASE B (CC OPTIMIZED)")
print("="*70)

phaseB_file = COOKED / 'step_08_phaseB_cc' / 'phaseB_summary.pkl'
phaseB = None

if phaseB_file.exists():
    with open(phaseB_file, 'rb') as f:
        phaseB = pickle.load(f)

    print("✓ Phase B (Classifier Chains - Optimized) loaded")
    if 'cc' in phaseB:
        phaseB_metrics = phaseB['cc']
        print(f"  Hamming Loss: {phaseB_metrics.get('hamming_loss', 'N/A'):.6f}")
        print(f"  Micro F1: {phaseB_metrics.get('micro_f1', 'N/A'):.4f}")
        print(f"  Macro F1: {phaseB_metrics.get('macro_f1', 'N/A'):.4f}")
else:
    print(f"✗ Phase B results not found at {phaseB_file}")

# Load Phase C Results (Label Powerset)
print("\n" + "="*70)
print("LOADING PHASE C (LABEL POWERSET)")
print("="*70)

phaseC_file = COOKED / 'step_09_phaseC_lp' / 'phaseC_summary.pkl'
phaseC = None

if phaseC_file.exists():
    with open(phaseC_file, 'rb') as f:
        phaseC = pickle.load(f)

    print("✓ Phase C (Label Powerset) loaded")
    if 'lp' in phaseC:
        phaseC_metrics = phaseC['lp']
        print(f"  Hamming Loss: {phaseC_metrics.get('hamming_loss', 'N/A'):.6f}")
        print(f"  Micro F1: {phaseC_metrics.get('micro_f1', 'N/A'):.4f}")
        print(f"  Macro F1: {phaseC_metrics.get('macro_f1', 'N/A'):.4f}")
else:
    print(f"✗ Phase C results not found at {phaseC_file}")

# Load Phase D Results (Deep Learning)
print("\n" + "="*70)
print("LOADING PHASE D (DEEP LEARNING)")
print("="*70)

PHASED_DIR = Path('/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration/analysis/outputs/phaseD')
architectures = ['mlp', 'cnn', 'cnn_lstm', 'transformer']
phaseD_results = {}

for arch in architectures:
    arch_dir = PHASED_DIR / arch
    summary_file = arch_dir / f'phaseD_summary_{arch}.pkl'

    if summary_file.exists():
        with open(summary_file, 'rb') as f:
            phaseD_results[arch] = pickle.load(f)

        best = phaseD_results[arch]['best']
        print(f"✓ Loaded {arch.upper():12}: Micro F1={best['metrics']['micro_f1']:.4f}, "
              f"Macro F1={best['metrics']['macro_f1']:.4f}")

if len(phaseD_results) > 0:
    print(f"\n✓ Loaded {len(phaseD_results)}/{len(architectures)} Phase D architectures")
else:
    print("\n✗ No Phase D results found")

# Build comprehensive comparison DataFrame
print("\n" + "="*70)
print("BUILDING COMPARISON TABLE")
print("="*70)

comparison_data = []

# Phase 2 (Baseline)
if phase2:
    comparison_data.append({
        'Phase': '2 (Baseline)',
        'Method': 'Multi-Output RF',
        'Type': 'Classical ML',
        'Approach': 'Algorithm Adaptation',
        'Hamming Loss': phase2['hamming_loss'],
        'Micro F1': phase2['micro_f1'],
        'Macro F1': phase2['macro_f1']
    })

# Phase A methods
if phaseA:
    if 'br' in phaseA:
        comparison_data.append({
            'Phase': 'A',
            'Method': 'Binary Relevance (BR)',
            'Type': 'Classical ML',
            'Approach': 'Problem Transformation',
            'Hamming Loss': phaseA['br'].get('hamming_loss'),
            'Micro F1': phaseA['br'].get('micro_f1'),
            'Macro F1': phaseA['br'].get('macro_f1')
        })

    if 'cc' in phaseA:
        comparison_data.append({
            'Phase': 'A',
            'Method': 'Classifier Chain (CC)',
            'Type': 'Classical ML',
            'Approach': 'Problem Transformation',
            'Hamming Loss': phaseA['cc'].get('hamming_loss'),
            'Micro F1': phaseA['cc'].get('micro_f1'),
            'Macro F1': phaseA['cc'].get('macro_f1')
        })

# Phase B
if phaseB and 'cc' in phaseB:
    comparison_data.append({
        'Phase': 'B',
        'Method': 'CC (Optimized)',
        'Type': 'Classical ML',
        'Approach': 'Problem Transformation',
        'Hamming Loss': phaseB['cc'].get('hamming_loss'),
        'Micro F1': phaseB['cc'].get('micro_f1'),
        'Macro F1': phaseB['cc'].get('macro_f1')
    })

# Phase C
if phaseC and 'lp' in phaseC:
    comparison_data.append({
        'Phase': 'C',
        'Method': 'Label Powerset (LP)',
        'Type': 'Classical ML',
        'Approach': 'Problem Transformation',
        'Hamming Loss': phaseC['lp'].get('hamming_loss'),
        'Micro F1': phaseC['lp'].get('micro_f1'),
        'Macro F1': phaseC['lp'].get('macro_f1')
    })

# Phase D
for arch, data in phaseD_results.items():
    best = data['best']['metrics']
    comparison_data.append({
        'Phase': 'D',
        'Method': f'{arch.upper().replace("_", "-")}',
        'Type': 'Deep Learning',
        'Approach': 'Algorithm Adaptation',
        'Hamming Loss': best.get('hamming'),
        'Micro F1': best.get('micro_f1'),
        'Macro F1': best.get('macro_f1')
    })

df_comparison = pd.DataFrame(comparison_data)

print("="*120)
print("COMPREHENSIVE MULTI-LABEL CLASSIFICATION COMPARISON: ALL PHASES")
print("="*120)
print(df_comparison.to_string(index=False))
print("="*120)

# Save comparison table
csv_path = OUT_DIR / 'all_phases_comparison.csv'
df_comparison.to_csv(csv_path, index=False)
print(f"\n✓ Saved comparison table to {csv_path}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY BY TYPE (Classical ML vs Deep Learning)")
print("="*80)
summary_type = df_comparison.groupby('Type')[['Hamming Loss', 'Micro F1', 'Macro F1']].agg(['mean', 'std', 'min', 'max'])
print(summary_type)

print("\n" + "="*80)
print("BEST METHODS PER METRIC")
print("="*80)

best_hamming = df_comparison.loc[df_comparison['Hamming Loss'].idxmin()]
print(f"Lowest Hamming Loss:  {best_hamming['Method']:25} (Phase {best_hamming['Phase']}) = {best_hamming['Hamming Loss']:.6f}")

best_micro = df_comparison.loc[df_comparison['Micro F1'].idxmax()]
print(f"Highest Micro F1:     {best_micro['Method']:25} (Phase {best_micro['Phase']}) = {best_micro['Micro F1']:.4f}")

best_macro = df_comparison.loc[df_comparison['Macro F1'].idxmax()]
print(f"Highest Macro F1:     {best_macro['Method']:25} (Phase {best_macro['Phase']}) = {best_macro['Macro F1']:.4f}")

# Visualization 1: Performance Comparison Bar Charts
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
metrics = ['Hamming Loss', 'Micro F1', 'Macro F1']
colors_map = {'Classical ML': 'steelblue', 'Deep Learning': 'darkorange'}

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    x_pos = np.arange(len(df_comparison))
    colors = [colors_map[t] for t in df_comparison['Type']]

    bars = ax.bar(x_pos, df_comparison[metric], color=colors, alpha=0.8, edgecolor='black')

    # Highlight best performer
    if metric == 'Hamming Loss':
        best_idx = df_comparison[metric].idxmin()
    else:
        best_idx = df_comparison[metric].idxmax()
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_comparison[metric])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=8, rotation=0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{row['Method']}\n(P{row['Phase']})" for _, row in df_comparison.iterrows()],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add reference lines
    if metric == 'Hamming Loss':
        ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Target (<0.05)')
    elif metric == 'Micro F1':
        ax.axhline(y=0.70, color='green', linestyle='--', alpha=0.5, label='Target (>0.70)')
    elif metric == 'Macro F1':
        ax.axhline(y=0.40, color='green', linestyle='--', alpha=0.5, label='Target (>0.40)')
    ax.legend()

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors_map[t], label=t, alpha=0.8) for t in colors_map.keys()]
fig.legend(handles=legend_elements, loc='upper center', ncol=2,
           bbox_to_anchor=(0.5, 0.98), frameon=True, fontsize=11)

plt.suptitle('Multi-Label Classification: Comprehensive Comparison (Phases A, B, C, D)',
             fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
output_file = OUT_DIR / 'all_phases_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_file}")

# Visualization 2: Radar Chart
top5 = df_comparison.nlargest(5, 'Micro F1')
top5_normalized = top5.copy()
top5_normalized['Hamming Accuracy'] = 1 - top5_normalized['Hamming Loss']
metrics_radar = ['Micro F1', 'Macro F1', 'Hamming Accuracy']

num_vars = len(metrics_radar)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

colors_list = ['steelblue', 'darkorange', 'green', 'red', 'purple']
for idx, (_, row) in enumerate(top5_normalized.iterrows()):
    values = [row[m] for m in metrics_radar]
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, label=f"{row['Method']} (P{row['Phase']})",
            color=colors_list[idx])
    ax.fill(angles, values, alpha=0.15, color=colors_list[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_radar, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.title('Top 5 Methods: Multi-Metric Radar Chart', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
output_file = OUT_DIR / 'radar_chart_top5.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_file}")

# Visualization 3: Phase Evolution
phase_best = df_comparison.loc[df_comparison.groupby('Phase')['Micro F1'].idxmax()]

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(phase_best))
width = 0.25

bars1 = ax.bar(x - width, phase_best['Hamming Loss'], width, label='Hamming Loss', color='tab:red', alpha=0.8)
bars2 = ax.bar(x, phase_best['Micro F1'], width, label='Micro F1', color='tab:blue', alpha=0.8)
bars3 = ax.bar(x + width, phase_best['Macro F1'], width, label='Macro F1', color='tab:green', alpha=0.8)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Phase (Best Method)', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Best Method per Phase: Performance Evolution', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([f"Phase {row['Phase']}\n{row['Method']}" for _, row in phase_best.iterrows()],
                   fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_file = OUT_DIR / 'phase_evolution.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_file}")

# Export comprehensive summary
export_summary = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'phases_analyzed': sorted(df_comparison['Phase'].unique().tolist()),
    'total_methods': len(df_comparison),
    'comparison_table': df_comparison.to_dict(),
    'best_overall': {
        'method': best_micro['Method'],
        'phase': best_micro['Phase'],
        'type': best_micro['Type'],
        'metrics': {
            'hamming_loss': best_micro['Hamming Loss'],
            'micro_f1': best_micro['Micro F1'],
            'macro_f1': best_micro['Macro F1']
        }
    }
}

summary_pkl = OUT_DIR / 'comprehensive_summary.pkl'
with open(summary_pkl, 'wb') as f:
    pickle.dump(export_summary, f)

print(f"✓ Saved comprehensive summary: {summary_pkl}")

print("\n" + "="*70)
print("NOTEBOOK COMPLETE")
print("="*70)
print(f"\nAll results saved to: {OUT_DIR}")
print(f"\nFiles generated:")
print(f"  • all_phases_comparison.csv")
print(f"  • all_phases_comparison.png")
print(f"  • radar_chart_top5.png")
print(f"  • phase_evolution.png")
print(f"  • comprehensive_summary.pkl")
print("\n✓ All outputs generated successfully")
