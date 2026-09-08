#!/usr/bin/env python
"""
Execute notebook 02 cells: Signal Visualization
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
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

print("="*70)
print("NOTEBOOK 02: SIGNAL VISUALIZATION")
print("="*70)
print("\n✓ Imports complete\n")

# Define data path
DATA_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')
FEATURES_FILE = DATA_DIR / 'features_engineered.pkl'
OUTPUT_DIR = DATA_DIR

print(f"Loading data from: {FEATURES_FILE}")
print(f"File size: {FEATURES_FILE.stat().st_size / (1024**3):.2f} GB")

# Load data
print("Loading pickle file...")
with open(FEATURES_FILE, 'rb') as f:
    data = pickle.load(f)

# Extract components
signal_names = data['signal_names']
sequences_full = data.get('sequences_full', None)
sequences_downsampled = data.get('sequences_downsampled', None)
y_binary = data['y_binary']

print(f"\n✓ Dataset loaded successfully")
print(f"  Total events: {len(y_binary):,}")
print(f"  Signals: {len(signal_names)}")
print(f"  Full sequences shape: {sequences_full.shape if sequences_full is not None else 'N/A'}")
print(f"  Downsampled sequences shape: {sequences_downsampled.shape if sequences_downsampled is not None else 'N/A'}")

# Use downsampled sequences for visualization (lighter)
if sequences_downsampled is not None:
    sequences = sequences_downsampled
    print(f"\nUsing downsampled sequences for visualization")
else:
    sequences = sequences_full
    print(f"\nUsing full sequences for visualization")

n_events, n_samples, n_signals = sequences.shape

print("\n" + "="*70)
print("SIGNAL VISUALIZATION")
print("="*70)
print(f"  Events: {n_events:,}")
print(f"  Time samples per event: {n_samples}")
print(f"  Signals per event: {n_signals}")

# Select representative events
normal_indices = np.where(y_binary == 0)[0]
fault_indices = np.where(y_binary == 1)[0]

# Pick 3 examples each
np.random.seed(42)
normal_examples = np.random.choice(normal_indices, min(3, len(normal_indices)), replace=False)
fault_examples = np.random.choice(fault_indices, min(3, len(fault_indices)), replace=False)

print(f"\nSelected examples:")
print(f"  Normal events: {normal_examples}")
print(f"  Fault events: {fault_examples}")

# Visualize signals for each example
time = np.arange(n_samples)

# Plot 1: Normal event example
fig, axes = plt.subplots(6, 1, figsize=(16, 12), sharex=True)
event_idx = normal_examples[0]

for i in range(min(6, n_signals)):
    signal_data = sequences[event_idx, :, i]
    axes[i].plot(time, signal_data, linewidth=0.8, alpha=0.9, color='steelblue')
    axes[i].set_ylabel(f"{signal_names[i][:25]}", fontsize=9)
    axes[i].grid(True, alpha=0.3)
    axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)

    # Add statistics
    mean_val = signal_data.mean()
    std_val = signal_data.std()
    axes[i].text(0.02, 0.95, f'μ={mean_val:.2e}, σ={std_val:.2e}',
                transform=axes[i].transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

axes[-1].set_xlabel('Sample Index', fontsize=11)
plt.suptitle(f'Normal Event Example (Index {event_idx})', fontsize=14, fontweight='bold')
plt.tight_layout()
output_file = OUTPUT_DIR / '02_signal_example_normal.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: {output_file}")

# Plot 2: Fault event example
fig, axes = plt.subplots(6, 1, figsize=(16, 12), sharex=True)
event_idx = fault_examples[0]

for i in range(min(6, n_signals)):
    signal_data = sequences[event_idx, :, i]
    axes[i].plot(time, signal_data, linewidth=0.8, alpha=0.9, color='coral')
    axes[i].set_ylabel(f"{signal_names[i][:25]}", fontsize=9)
    axes[i].grid(True, alpha=0.3)
    axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)

    # Add statistics
    mean_val = signal_data.mean()
    std_val = signal_data.std()
    axes[i].text(0.02, 0.95, f'μ={mean_val:.2e}, σ={std_val:.2e}',
                transform=axes[i].transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

axes[-1].set_xlabel('Sample Index', fontsize=11)
plt.suptitle(f'Fault Event Example (Index {event_idx})', fontsize=14, fontweight='bold')
plt.tight_layout()
output_file = OUTPUT_DIR / '02_signal_example_fault.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_file}")

# Plot 3: Comparison of normal vs fault for key signals
key_signals = [0, 1, 2, 3, 4, 5]  # First 6 signals
n_compare = min(6, len(key_signals))

fig, axes = plt.subplots(n_compare, 2, figsize=(18, 3*n_compare))

for i, sig_idx in enumerate(key_signals[:n_compare]):
    # Normal example
    axes[i, 0].plot(time, sequences[normal_examples[0], :, sig_idx],
                   linewidth=0.8, alpha=0.9, color='steelblue', label='Normal')
    axes[i, 0].set_ylabel(f"{signal_names[sig_idx][:20]}", fontsize=9)
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 0].legend(loc='upper right')
    if i == 0:
        axes[i, 0].set_title('Normal Event', fontsize=12, fontweight='bold')

    # Fault example
    axes[i, 1].plot(time, sequences[fault_examples[0], :, sig_idx],
                   linewidth=0.8, alpha=0.9, color='coral', label='Fault')
    axes[i, 1].set_ylabel(f"{signal_names[sig_idx][:20]}", fontsize=9)
    axes[i, 1].grid(True, alpha=0.3)
    axes[i, 1].legend(loc='upper right')
    if i == 0:
        axes[i, 1].set_title('Fault Event', fontsize=12, fontweight='bold')

axes[-1, 0].set_xlabel('Sample Index', fontsize=11)
axes[-1, 1].set_xlabel('Sample Index', fontsize=11)

plt.suptitle('Normal vs Fault Signal Comparison', fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
output_file = OUTPUT_DIR / '02_signal_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_file}")

# Plot 4: Signal statistics across all events
print(f"\n" + "="*70)
print("COMPUTING SIGNAL STATISTICS")
print("="*70)

signal_means = sequences.mean(axis=(0, 1))  # Mean across events and time
signal_stds = sequences.std(axis=(0, 1))    # Std across events and time
signal_mins = sequences.min(axis=(0, 1))
signal_maxs = sequences.max(axis=(0, 1))

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Mean
axes[0, 0].bar(range(n_signals), signal_means, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Signal Index')
axes[0, 0].set_ylabel('Mean Value')
axes[0, 0].set_title('Signal Means (across all events & time)', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Std
axes[0, 1].bar(range(n_signals), signal_stds, color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Signal Index')
axes[0, 1].set_ylabel('Std Dev')
axes[0, 1].set_title('Signal Standard Deviations', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Min/Max ranges
ranges = signal_maxs - signal_mins
axes[1, 0].bar(range(n_signals), ranges, color='green', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Signal Index')
axes[1, 0].set_ylabel('Range (Max - Min)')
axes[1, 0].set_title('Signal Ranges', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Coefficient of variation (CV = std/mean)
cv = np.abs(signal_stds / (signal_means + 1e-10))
axes[1, 1].bar(range(n_signals), cv, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Signal Index')
axes[1, 1].set_ylabel('Coefficient of Variation')
axes[1, 1].set_title('Signal Variability (CV = σ/|μ|)', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].set_yscale('log')

plt.tight_layout()
output_file = OUTPUT_DIR / '02_signal_statistics.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_file}")

# Statistics by class
print(f"\n" + "="*70)
print("SIGNAL STATISTICS BY CLASS")
print("="*70)

normal_sequences = sequences[y_binary == 0]
fault_sequences = sequences[y_binary == 1]

normal_means = normal_sequences.mean(axis=(0, 1))
fault_means = fault_sequences.mean(axis=(0, 1))
normal_stds = normal_sequences.std(axis=(0, 1))
fault_stds = fault_sequences.std(axis=(0, 1))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Mean comparison
x_pos = np.arange(n_signals)
width = 0.35
axes[0].bar(x_pos - width/2, normal_means, width, label='Normal', color='steelblue', alpha=0.7, edgecolor='black')
axes[0].bar(x_pos + width/2, fault_means, width, label='Fault', color='coral', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Signal Index', fontsize=11)
axes[0].set_ylabel('Mean Value', fontsize=11)
axes[0].set_title('Signal Means by Class', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)

# Std comparison
axes[1].bar(x_pos - width/2, normal_stds, width, label='Normal', color='steelblue', alpha=0.7, edgecolor='black')
axes[1].bar(x_pos + width/2, fault_stds, width, label='Fault', color='coral', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Signal Index', fontsize=11)
axes[1].set_ylabel('Std Dev', fontsize=11)
axes[1].set_title('Signal Std Deviations by Class', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_file = OUTPUT_DIR / '02_signal_statistics_by_class.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_file}")

print(f"\nMost discriminative signals (by mean difference):")
mean_diffs = np.abs(normal_means - fault_means)
top_indices = np.argsort(mean_diffs)[::-1][:10]
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank:2d}. Signal {idx:2d} ({signal_names[idx][:30]}): Δμ = {mean_diffs[idx]:.3e}")

print("\n" + "="*70)
print("NOTEBOOK 02 COMPLETE")
print("="*70)
print(f"\nKey Results:")
print(f"  - Visualized {n_signals} signals across {n_events:,} events")
print(f"  - Generated example plots for normal and fault events")
print(f"  - Analyzed signal statistics by class")
print(f"  - All visualizations saved to {OUTPUT_DIR}")
print("\n✓ All outputs generated successfully")
