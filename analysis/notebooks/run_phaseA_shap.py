#!/usr/bin/env python
"""
Execute phaseA_shap_analysis.ipynb cells - SHAP Analysis for Binary Relevance Multi-Label Classification
Loads Phase A results (BR + CC) and generates SHAP feature importance plots
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
import shap
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (14, 6)

print("="*70)
print("PHASE A: BINARY RELEVANCE SHAP ANALYSIS")
print("="*70)
print("\n✓ Imports complete\n")

# Paths
COOKED = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')
PHASEA = COOKED / 'step_07_phaseA_br'
SHAP_FILE = PHASEA / 'shap_br.npz'
BR_FILE = PHASEA / 'br_results.pkl'
SUMMARY_FILE = PHASEA / 'phaseA_summary.pkl'
OUT_DIR = Path('analysis/outputs/phaseA_shap')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('Phase A Results Directory:', PHASEA)
print('SHAP file:', SHAP_FILE.exists())
print('BR model:', BR_FILE.exists())
print('Summary:', SUMMARY_FILE.exists())
print('Notebook outputs:', OUT_DIR)

# Load summary and model
print("\n" + "="*70)
print("LOADING PHASE A RESULTS")
print("="*70)

with open(SUMMARY_FILE, 'rb') as f:
    summary = pickle.load(f)

print('Summary keys:', list(summary.keys()))

# Extract label names
label_names = summary.get('label_names', [])
print(f'\nLabel Names ({len(label_names)} total):')
for i, name in enumerate(label_names, 1):
    print(f'  {i}. {name}')

# Compare BR vs CC metrics
br = summary.get('br')
cc = summary.get('cc')

rows = []
if br:
    rows.append({
        'model': 'Binary Relevance (BR)',
        'hamming': br.get('hamming_loss'),
        'micro_f1': br.get('micro_f1'),
        'macro_f1': br.get('macro_f1')
    })
if cc:
    rows.append({
        'model': 'Classifier Chain (CC)',
        'hamming': cc.get('hamming_loss'),
        'micro_f1': cc.get('micro_f1'),
        'macro_f1': cc.get('macro_f1')
    })

df_comparison = pd.DataFrame(rows).set_index('model')

print("\n" + "="*80)
print("PHASE A: BINARY RELEVANCE VS CLASSIFIER CHAIN COMPARISON")
print("="*80)
print(df_comparison.to_string())
print("="*80)

if br and cc:
    print("\nImprovement (CC over BR):")
    print(f"  Hamming Loss: {(br['hamming_loss'] - cc['hamming_loss'])/br['hamming_loss']*100:+.2f}%")
    print(f"  Micro F1:     {(cc['micro_f1'] - br['micro_f1'])/br['micro_f1']*100:+.2f}%")
    print(f"  Macro F1:     {(cc['macro_f1'] - br['macro_f1'])/br['macro_f1']*100:+.2f}%")

# Save comparison
csv_path = OUT_DIR / 'phaseA_metrics_comparison.csv'
df_comparison.to_csv(csv_path)
print(f"\n✓ Saved metrics to {csv_path}")

# Visualize BR vs CC comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = ['hamming', 'micro_f1', 'macro_f1']
titles = ['Hamming Loss\n(lower is better)', 'Micro F1\n(higher is better)', 'Macro F1\n(higher is better)']
colors = ['tab:red', 'tab:blue', 'tab:green']

for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
    ax = axes[idx]
    values = df_comparison[metric].values
    models = df_comparison.index.tolist()

    bars = ax.bar(models, values, color=color, alpha=0.7, edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
    ax.set_ylim([0, max(values) * 1.15])
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Phase A: Binary Relevance vs Classifier Chain Performance', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
output_file = OUT_DIR / 'phaseA_br_cc_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_file}")

# Per-label performance analysis
print("\n" + "="*70)
print("PER-LABEL PERFORMANCE ANALYSIS")
print("="*70)

per_label_data = []

for i, label_name in enumerate(label_names):
    row = {'Fault Type': label_name}

    if br and 'per_label' in br:
        br_report = br['per_label'].get(label_name, {})
        br_f1 = br_report.get('1', {}).get('f1-score', 0.0) if '1' in br_report else br_report.get('weighted avg', {}).get('f1-score', 0.0)
        row['BR F1'] = br_f1

    if cc and 'per_label' in cc:
        cc_report = cc['per_label'].get(label_name, {})
        cc_f1 = cc_report.get('1', {}).get('f1-score', 0.0) if '1' in cc_report else cc_report.get('weighted avg', {}).get('f1-score', 0.0)
        row['CC F1'] = cc_f1

    if 'BR F1' in row and 'CC F1' in row:
        row['Improvement'] = row['CC F1'] - row['BR F1']

    per_label_data.append(row)

df_per_label = pd.DataFrame(per_label_data)

print("="*100)
print(df_per_label.to_string(index=False))
print("="*100)

df_per_label.to_csv(OUT_DIR / 'phaseA_per_label_comparison.csv', index=False)
print(f"\n✓ Saved per-label comparison to {OUT_DIR / 'phaseA_per_label_comparison.csv'}")

# Visualize per-label F1 scores
if 'BR F1' in df_per_label.columns and 'CC F1' in df_per_label.columns:
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(label_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, df_per_label['BR F1'], width, label='Binary Relevance', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, df_per_label['CC F1'], width, label='Classifier Chain', color='darkorange', edgecolor='black')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Fault Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Label F1-Score: BR vs CC', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([name[:30] + '...' if len(name) > 30 else name for name in label_names], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    output_file = OUT_DIR / 'phaseA_per_label_f1.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

# Load SHAP values
print("\n" + "="*70)
print("LOADING SHAP VALUES")
print("="*70)

sh = np.load(SHAP_FILE, allow_pickle=True)
print('SHAP file keys:', list(sh.keys()))

# Load feature names
feature_names = None
if 'feature_names' in sh:
    feature_names = list(sh['feature_names'])
    print(f'✓ Feature names loaded from SHAP file ({len(feature_names)} features)')
else:
    # Fallback: try features artifact
    for cand in [COOKED / 'step_03_features' / 'features_engineered.pkl', COOKED / 'features_engineered.pkl']:
        if cand.exists():
            with open(cand, 'rb') as f:
                feat = pickle.load(f)
            feature_names = feat.get('feature_names') or feat.get('feature_cols')
            if feature_names is not None:
                feature_names = list(feature_names)
                print(f'✓ Loaded feature names from {cand.name} ({len(feature_names)} features)')
                break

if feature_names is None:
    print('⚠️  Feature names not found, using generic labels')
    for key in sh.keys():
        if key.startswith('shap_'):
            n_features = sh[key].shape[1] if len(sh[key].shape) >= 2 else 1243
            feature_names = [f'feat_{i}' for i in range(n_features)]
            break

print(f'\nTotal features: {len(feature_names) if feature_names else "unknown"}')

# Generate SHAP plots
print("\n" + "="*70)
print("GENERATING SHAP VISUALIZATIONS")
print("="*70)

shap.initjs()

for i, lbl in enumerate(label_names):
    key = f'shap_{i}'
    if key not in sh:
        print(f'⚠️  No SHAP values for {lbl} (key: {key})')
        continue

    shap_vals = sh[key]

    # Handle multi-class SHAP values
    if len(shap_vals.shape) == 3:
        shap_vals_plot = shap_vals[:, :, -1]
    else:
        shap_vals_plot = shap_vals

    # SHAP summary plot
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals_plot, features=None, feature_names=feature_names, show=False, max_display=20)
        plt.title(f'SHAP Summary: {lbl}', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        output_file = OUT_DIR / f'shap_summary_label_{i}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f'⚠️  SHAP summary plot failed for {lbl}: {e}')

    # Top-20 mean |SHAP| bar chart
    try:
        mean_abs = np.mean(np.abs(shap_vals_plot), axis=0)
        idx = np.argsort(mean_abs)[-20:]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(idx)), mean_abs[idx], color='steelblue', edgecolor='black')
        ax.set_yticks(range(len(idx)))
        names = [(feature_names[j] if feature_names is not None else f'feat_{j}') for j in idx]
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
        ax.set_title(f'Top 20 Features by SHAP Importance\n{lbl}', fontsize=13, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        output_file = OUT_DIR / f'top20_label_{i}.png'
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'⚠️  Top-20 plot failed for {lbl}: {e}')

    print(f'✓ Saved SHAP plots for: {lbl}')

print(f'\n✓ All SHAP plots saved in {OUT_DIR}')

# Export summary
export_summary = {
    'phase': 'A',
    'methods': ['Binary Relevance', 'Classifier Chain'],
    'metrics': df_comparison.to_dict(),
    'per_label': df_per_label.to_dict() if not df_per_label.empty else None,
    'best_method': 'Classifier Chain' if cc and cc.get('macro_f1', 0) > br.get('macro_f1', 0) else 'Binary Relevance',
    'dataset': {
        'n_labels': len(label_names),
        'n_features': len(feature_names) if feature_names else 1243,
        'label_names': label_names
    }
}

export_path = OUT_DIR / 'phaseA_export_summary.pkl'
with open(export_path, 'wb') as f:
    pickle.dump(export_summary, f)

print(f"\n✓ Exported Phase A summary to {export_path}")
print("\nExport Summary:")
print(f"  Phase: {export_summary['phase']}")
print(f"  Methods: {export_summary['methods']}")
print(f"  Best Method: {export_summary['best_method']}")
print(f"  Dataset: {export_summary['dataset']['n_labels']} labels, {export_summary['dataset']['n_features']} features")

print("\n" + "="*70)
print("NOTEBOOK COMPLETE")
print("="*70)
print(f"\nAll results saved to: {OUT_DIR}")
print(f"\nFiles generated:")
print(f"  • phaseA_metrics_comparison.csv")
print(f"  • phaseA_br_cc_comparison.png")
print(f"  • phaseA_per_label_comparison.csv")
print(f"  • phaseA_per_label_f1.png")
print(f"  • shap_summary_label_*.png (for each label)")
print(f"  • top20_label_*.png (for each label)")
print(f"  • phaseA_export_summary.pkl")
print("\n✓ All outputs generated successfully")
