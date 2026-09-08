#!/usr/bin/env python
"""
Execute phaseD_deep_analysis.ipynb cells - Deep Learning Multi-Head Analysis
Loads Phase D results (MLP, CNN, CNN-LSTM, Transformer) and compares with classical methods
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
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (14, 6)

print("="*70)
print("PHASE D: DEEP LEARNING MULTI-HEAD ANALYSIS")
print("="*70)
print("\n✓ Imports complete\n")

# Paths
COOKED = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')
OUTDIR = Path('/pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration/analysis/outputs/phaseD')
NOTEBOOK_OUT = Path('analysis/outputs/phaseD_analysis')
NOTEBOOK_OUT.mkdir(parents=True, exist_ok=True)

architectures = ['mlp', 'cnn', 'cnn_lstm', 'transformer']

print('Phase D Results Directory:', OUTDIR)
print('Notebook outputs:', NOTEBOOK_OUT)
print('\nArchitectures to analyze:', architectures)

# Load training results from all architectures
print("\n" + "="*70)
print("LOADING PHASE D RESULTS")
print("="*70)

results = {}
missing = []

for arch in architectures:
    arch_dir = OUTDIR / arch
    summary_file = arch_dir / f'phaseD_summary_{arch}.pkl'

    if summary_file.exists():
        with open(summary_file, 'rb') as f:
            results[arch] = pickle.load(f)

        best = results[arch]['best']
        print(f"✓ Loaded {arch.upper():12} - Best Epoch: {best['epoch']:3d} | "
              f"Micro F1: {best['metrics']['micro_f1']:.4f} | "
              f"Macro F1: {best['metrics']['macro_f1']:.4f} | "
              f"Hamming: {best['metrics']['hamming']:.6f}")
    else:
        missing.append(arch)
        print(f"✗ Missing {arch.upper():12} - {summary_file}")

print(f"\n{'='*80}")
print(f"Successfully loaded: {len(results)}/{len(architectures)} architectures")
if missing:
    print(f"Missing architectures: {missing}")
print(f"{'='*80}")

# Create performance comparison table
print("\n" + "="*70)
print("BUILDING COMPARISON TABLE")
print("="*70)

comparison_data = []

for arch in architectures:
    if arch in results:
        best = results[arch]['best']['metrics']
        comparison_data.append({
            'Architecture': arch.upper().replace('_', '-'),
            'Hamming Loss': best['hamming'],
            'Micro F1': best['micro_f1'],
            'Macro F1': best['macro_f1'],
            'Best Epoch': results[arch]['best']['epoch']
        })

df_comparison = pd.DataFrame(comparison_data)

print("="*100)
print("PHASE D: DEEP LEARNING MULTI-HEAD ARCHITECTURE COMPARISON")
print("="*100)
print(df_comparison.to_string(index=False))
print("="*100)

if len(df_comparison) > 0:
    print("\nBest Architecture per Metric:")
    print(f"  • Lowest Hamming Loss:  {df_comparison.loc[df_comparison['Hamming Loss'].idxmin(), 'Architecture']} "
          f"({df_comparison['Hamming Loss'].min():.6f})")
    print(f"  • Highest Micro F1:     {df_comparison.loc[df_comparison['Micro F1'].idxmax(), 'Architecture']} "
          f"({df_comparison['Micro F1'].max():.4f})")
    print(f"  • Highest Macro F1:     {df_comparison.loc[df_comparison['Macro F1'].idxmax(), 'Architecture']} "
          f"({df_comparison['Macro F1'].max():.4f})")

    # Save comparison
    csv_path = NOTEBOOK_OUT / 'phaseD_comparison.csv'
    df_comparison.to_csv(csv_path, index=False)
    print(f"\n✓ Saved comparison table to {csv_path}")

# Visualize performance comparison
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

if len(df_comparison) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['Hamming Loss', 'Micro F1', 'Macro F1']
    colors = ['tab:red', 'tab:blue', 'tab:green']
    invert = [True, False, False]  # Hamming loss: lower is better

    for idx, (metric, color, inv) in enumerate(zip(metrics, colors, invert)):
        ax = axes[idx]
        bars = ax.bar(df_comparison['Architecture'], df_comparison[metric],
                      color=color, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom' if not inv else 'top',
                    fontsize=10, fontweight='bold')

        # Highlight best performer
        if inv:
            best_idx = df_comparison[metric].idxmin()
        else:
            best_idx = df_comparison[metric].idxmax()
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

        ax.set_title(f'{metric}\n({"lower" if inv else "higher"} is better)',
                     fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xlabel('Architecture', fontsize=11)
        ax.tick_params(axis='x', rotation=0)
        ax.grid(axis='y', alpha=0.3)

        # Add reference line
        if metric == 'Hamming Loss':
            ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Target (0.05)')
        elif metric == 'Micro F1':
            ax.axhline(y=0.70, color='green', linestyle='--', alpha=0.5, label='Target (0.70)')
        elif metric == 'Macro F1':
            ax.axhline(y=0.40, color='green', linestyle='--', alpha=0.5, label='Target (0.40)')
        ax.legend()

    plt.suptitle('Phase D: Deep Learning Architecture Performance Comparison',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_file = NOTEBOOK_OUT / 'phaseD_architecture_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

# Training efficiency analysis
print("\n" + "="*80)
print("TRAINING EFFICIENCY ANALYSIS")
print("="*80)

for arch in architectures:
    if arch in results:
        best_epoch = results[arch]['best']['epoch']
        total_epochs = results[arch]['config'].get('epochs', 50)
        print(f"{arch.upper():12} - Converged at epoch {best_epoch}/{total_epochs} "
              f"({best_epoch/total_epochs*100:.1f}% of training)")

print("="*80)

# Architecture-specific details
print("\n" + "="*70)
print("ARCHITECTURE-SPECIFIC DETAILS")
print("="*70)

for arch in architectures:
    if arch not in results:
        continue

    print("\n" + "="*80)
    print(f"ARCHITECTURE: {arch.upper().replace('_', '-')}")
    print("="*80)

    config = results[arch]['config']
    best = results[arch]['best']
    final = results[arch].get('final', best)

    print("\nConfiguration:")
    print(f"  • Encoder: {arch.upper()}")
    print(f"  • Bottleneck dimension: {config.get('bottleneck', 256)}")
    print(f"  • Batch size: {config.get('batch_size', 128)}")
    print(f"  • Learning rate: {config.get('lr', 0.001)}")
    print(f"  • Device: {config.get('device', 'cpu')}")

    print("\nBest Validation Performance:")
    print(f"  • Epoch: {best['epoch']}")
    print(f"  • Hamming Loss: {best['metrics']['hamming']:.6f}")
    print(f"  • Micro F1: {best['metrics']['micro_f1']:.4f}")
    print(f"  • Macro F1: {best['metrics']['macro_f1']:.4f}")

    if final != best:
        print("\nFinal Performance (last epoch):")
        print(f"  • Hamming Loss: {final['hamming']:.6f}")
        print(f"  • Micro F1: {final['micro_f1']:.4f}")
        print(f"  • Macro F1: {final['macro_f1']:.4f}")

        if final['micro_f1'] < best['metrics']['micro_f1'] - 0.02:
            print("\n⚠️  Warning: Significant performance drop from best → final suggests overfitting")

print("\n" + "="*80)

# Load classical method results for comparison
print("\n" + "="*70)
print("LOADING CLASSICAL METHODS FOR COMPARISON")
print("="*70)

classical_results = {}

# Phase A
phaseA_path = Path('analysis/outputs/phaseA_shap/phaseA_export_summary.pkl')
if phaseA_path.exists():
    with open(phaseA_path, 'rb') as f:
        classical_results['Phase A'] = pickle.load(f)
    print("✓ Loaded Phase A results (Binary Relevance + Classifier Chain)")
else:
    print(f"⚠️  Phase A results not found at {phaseA_path}")

# Phase B
phaseB_summary = COOKED / 'step_08_phaseB_cc' / 'phaseB_summary.pkl'
if phaseB_summary.exists():
    with open(phaseB_summary, 'rb') as f:
        classical_results['Phase B (CC)'] = pickle.load(f)
    print("✓ Loaded Phase B results (Classifier Chains)")

# Phase C
phaseC_summary = COOKED / 'step_09_phaseC_lp' / 'phaseC_summary.pkl'
if phaseC_summary.exists():
    with open(phaseC_summary, 'rb') as f:
        classical_results['Phase C (LP)'] = pickle.load(f)
    print("✓ Loaded Phase C results (Label Powerset)")

print(f"\nClassical methods loaded: {len(classical_results)}")

# Build comprehensive comparison
print("\n" + "="*70)
print("CROSS-PHASE COMPARISON")
print("="*70)

all_methods = []

# Add classical methods
if 'Phase A' in classical_results:
    metrics = classical_results['Phase A']['metrics']
    if 'hamming' in metrics:
        for method, vals in metrics['hamming'].items():
            all_methods.append({
                'Phase': 'A',
                'Method': method.replace('Binary Relevance (BR)', 'BR').replace('Classifier Chain (CC)', 'CC'),
                'Type': 'Classical ML',
                'Hamming Loss': metrics['hamming'][method],
                'Micro F1': metrics['micro_f1'][method],
                'Macro F1': metrics['macro_f1'][method]
            })

# Add deep learning methods
if len(df_comparison) > 0:
    for _, row in df_comparison.iterrows():
        all_methods.append({
            'Phase': 'D',
            'Method': row['Architecture'],
            'Type': 'Deep Learning',
            'Hamming Loss': row['Hamming Loss'],
            'Micro F1': row['Micro F1'],
            'Macro F1': row['Macro F1']
        })

if len(all_methods) > 0:
    df_all = pd.DataFrame(all_methods)

    print("="*100)
    print("COMPREHENSIVE COMPARISON: CLASSICAL ML vs DEEP LEARNING")
    print("="*100)
    print(df_all.to_string(index=False))
    print("="*100)

    print("\nSummary by Type:")
    print(df_all.groupby('Type')[['Hamming Loss', 'Micro F1', 'Macro F1']].agg(['mean', 'std', 'min', 'max']))

    # Save comprehensive comparison
    df_all.to_csv(NOTEBOOK_OUT / 'all_phases_comparison.csv', index=False)
    print(f"\n✓ Saved comprehensive comparison to {NOTEBOOK_OUT / 'all_phases_comparison.csv'}")

    # Visualize cross-phase comparison
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metrics_to_plot = ['Hamming Loss', 'Micro F1', 'Macro F1']
    colors_map = {'Classical ML': 'steelblue', 'Deep Learning': 'darkorange'}

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Group by type
        for method_type in ['Classical ML', 'Deep Learning']:
            subset = df_all[df_all['Type'] == method_type]
            if len(subset) > 0:
                x_pos = np.arange(len(subset))
                offset = 0.2 if method_type == 'Classical ML' else -0.2
                ax.bar(x_pos + offset, subset[metric], width=0.35,
                       label=method_type, color=colors_map[method_type],
                       alpha=0.8, edgecolor='black')

        ax.set_xticks(np.arange(len(df_all)))
        ax.set_xticklabels(df_all['Method'], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Classical ML vs Deep Learning: Multi-Label Classification Performance',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_file = NOTEBOOK_OUT / 'classical_vs_deep_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS FOR PRODUCTION DEPLOYMENT")
print("="*80)

if len(df_comparison) > 0:
    best_arch = df_comparison.loc[df_comparison['Micro F1'].idxmax()]

    print(f"\n1. Best Performing Architecture: {best_arch['Architecture']}")
    print(f"   • Micro F1: {best_arch['Micro F1']:.4f}")
    print(f"   • Macro F1: {best_arch['Macro F1']:.4f}")
    print(f"   • Hamming Loss: {best_arch['Hamming Loss']:.6f}")

    if best_arch['Micro F1'] >= 0.70:
        print("\n   ✓ MEETS performance target (Micro F1 ≥ 0.70)")
        print("   → Recommendation: DEPLOY this architecture for production")
    else:
        print("\n   ✗ BELOW performance target (Micro F1 < 0.70)")
        print("   → Recommendation: Further tuning or hybrid approach needed")

    print("\n2. Deployment Strategy:")
    if len(all_methods) > 0 and 'Type' in df_all.columns:
        best_classical = df_all[df_all['Type'] == 'Classical ML']['Micro F1'].max() if 'Classical ML' in df_all['Type'].values else 0
        best_deep = df_all[df_all['Type'] == 'Deep Learning']['Micro F1'].max() if 'Deep Learning' in df_all['Type'].values else 0

        if best_deep > best_classical + 0.03:
            print("   → Deep learning shows significant improvement (+3% Micro F1)")
            print("   → Deploy deep learning model")
        elif best_classical > best_deep + 0.01:
            print("   → Classical ML performs better or comparably")
            print("   → Use classical ML for better interpretability and speed")
        else:
            print("   → Performance is comparable between deep learning and classical ML")
            print("   → Decision factors:")
            print("       * Interpretability needed? → Use Classical ML (Phase A/B/C)")
            print("       * Complex patterns expected? → Use Deep Learning (Phase D)")

    print("\n3. Potential Improvements:")
    print("   • Hyperparameter tuning (learning rate, dropout, hidden dims)")
    print("   • Data augmentation for minority fault types")
    print("   • Ensemble of top-2 architectures")
    print("   • Class weighting in loss function")
    print("   • Feature engineering guided by SHAP analysis")
    print("   • Label-specific thresholds (instead of fixed 0.5)")

# Export results
if len(df_comparison) > 0:
    export_summary = {
        'phase': 'D',
        'methods': list(df_comparison['Architecture']),
        'metrics': df_comparison.to_dict(),
        'best_architecture': df_comparison.loc[df_comparison['Micro F1'].idxmax(), 'Architecture'],
        'comparison_with_classical': df_all.to_dict() if len(all_methods) > 0 else None,
        'dataset': {
            'n_labels': 7,
            'n_features': 1243
        }
    }

    export_path = NOTEBOOK_OUT / 'phaseD_export_summary.pkl'
    with open(export_path, 'wb') as f:
        pickle.dump(export_summary, f)

    print(f"\n✓ Exported Phase D summary to {export_path}")

print("\n" + "="*70)
print("NOTEBOOK COMPLETE")
print("="*70)
print(f"\nAll results saved to: {NOTEBOOK_OUT}")
print(f"\nFiles generated:")
print(f"  • phaseD_comparison.csv")
print(f"  • phaseD_architecture_comparison.png")
print(f"  • all_phases_comparison.csv")
print(f"  • classical_vs_deep_comparison.png")
print(f"  • phaseD_export_summary.pkl")
print("\n✓ All outputs generated successfully")
