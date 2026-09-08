#!/usr/bin/env python3
"""
Investigate the relationship between LOOP, ALM, and trigger types
to correctly identify manual vs automatic triggers
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add PyPostMortem to path
pypm_path = Path('/pbs/throng/m4cast/projects/SPIRAL2/programmes/PyPostMortem/src')
sys.path.insert(0, str(pypm_path))

from PyPostmortem.utils.PyPostMortem import Read_Signals

# Load the metadata that was already created
OUTPUT_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')
metadata_file = OUTPUT_DIR / 'metadata_sample_27_events.csv'

print("Loading metadata...")
df = pd.read_csv(metadata_file)

print(f"\nDataset: {len(df)} events")
print("="*80)

# Analyze LOOP vs ALM relationship
print("\n1. LOOP vs ALM Analysis")
print("-"*80)

# Create ALM categories
df['ALM_int'] = df['ALM'].apply(lambda x: int(x, 16) if isinstance(x, str) else x)
df['has_fault'] = df['ALM_int'] > 0

# Cross-tabulation
crosstab = pd.crosstab(
    df['LOOP'],
    df['has_fault'],
    rownames=['LOOP'],
    colnames=['Has Fault (ALM>0)'],
    margins=True
)
print(crosstab)

print("\n\n2. Detailed Breakdown")
print("-"*80)

# LOOP=OFF events
loop_off = df[df['LOOP'] == 'OFF']
print(f"\nLOOP=OFF events: {len(loop_off)}")
if len(loop_off) > 0:
    print(f"  With faults (ALM>0): {loop_off['has_fault'].sum()}")
    print(f"  Without faults (ALM=0): {(~loop_off['has_fault']).sum()}")
    print(f"\n  Files:")
    for idx, row in loop_off.iterrows():
        print(f"    - {row['Filename']}")
        print(f"      LOOP: {row['LOOP']}, ALM: {row['ALM']}, KPI: {row['KPI']}, NDEC: {row['NDEC']}")

# LOOP=ON events
loop_on = df[df['LOOP'] == 'ON']
print(f"\n\nLOOP=ON events: {len(loop_on)}")
if len(loop_on) > 0:
    print(f"  With faults (ALM>0): {loop_on['has_fault'].sum()}")
    print(f"  Without faults (ALM=0): {(~loop_on['has_fault']).sum()}")

    # Show a few examples of LOOP=ON with and without faults
    print(f"\n  Examples with faults:")
    with_faults = loop_on[loop_on['has_fault']].head(3)
    for idx, row in with_faults.iterrows():
        print(f"    - {row['Filename']}")
        print(f"      ALM: {row['ALM']}, Faults: {row['Nombre de défauts']}")

    print(f"\n  Examples without faults:")
    without_faults = loop_on[~loop_on['has_fault']].head(3)
    for idx, row in without_faults.iterrows():
        print(f"    - {row['Filename']}")
        print(f"      ALM: {row['ALM']}")

print("\n\n3. Hypothesis Testing")
print("-"*80)

print("\nCurrent interpretation (in notebook):")
print("  - LOOP=OFF → Manual trigger")
print("  - LOOP=ON → Automatic trigger")

print("\nAlternative interpretation 1:")
print("  - ALM=0 (no faults) → Manual acquisition (operator-initiated)")
print("  - ALM>0 (has faults) → Automatic trigger (fault-initiated)")

print("\nAlternative interpretation 2:")
print("  - LOOP=OFF + ALM=0 → Manual acquisition")
print("  - LOOP=ON + ALM>0 → Automatic fault trigger")
print("  - LOOP=ON + ALM=0 → Automatic acquisition (no fault detected)")
print("  - LOOP=OFF + ALM>0 → ??? (rare case)")

# Calculate statistics for each interpretation
print("\n\n4. Statistics for Different Interpretations")
print("-"*80)

print("\nInterpretation 1 (Current - based on LOOP):")
print(f"  Manual (LOOP=OFF): {len(loop_off)} ({100*len(loop_off)/len(df):.1f}%)")
print(f"  Automatic (LOOP=ON): {len(loop_on)} ({100*len(loop_on)/len(df):.1f}%)")

print("\nInterpretation 2 (Based on ALM):")
no_fault = df[~df['has_fault']]
with_fault = df[df['has_fault']]
print(f"  Manual/Normal (ALM=0): {len(no_fault)} ({100*len(no_fault)/len(df):.1f}%)")
print(f"  Automatic Fault (ALM>0): {len(with_fault)} ({100*len(with_fault)/len(df):.1f}%)")

print("\nInterpretation 3 (Combined):")
manual = df[(df['LOOP'] == 'OFF') & (~df['has_fault'])]
auto_fault = df[(df['LOOP'] == 'ON') & (df['has_fault'])]
auto_normal = df[(df['LOOP'] == 'ON') & (~df['has_fault'])]
rare = df[(df['LOOP'] == 'OFF') & (df['has_fault'])]

print(f"  Manual acquisition (LOOP=OFF + ALM=0): {len(manual)} ({100*len(manual)/len(df):.1f}%)")
print(f"  Automatic fault (LOOP=ON + ALM>0): {len(auto_fault)} ({100*len(auto_fault)/len(df):.1f}%)")
print(f"  Automatic normal (LOOP=ON + ALM=0): {len(auto_normal)} ({100*len(auto_normal)/len(df):.1f}%)")
print(f"  Rare case (LOOP=OFF + ALM>0): {len(rare)} ({100*len(rare)/len(df):.1f}%)")

print("\n\n5. Recommendation")
print("-"*80)
print("Based on the user's statement:")
print("  'postmortem data generated on trigger' = automatic (likely ALM>0)")
print("  'files generated manually (no ALM)' = manual (likely ALM=0)")
print("\nSuggested correction:")
print("  - Trigger type should be based primarily on ALM, not LOOP")
print("  - LOOP indicates control loop state (feedback ON/OFF)")
print("  - ALM indicates whether data was captured due to fault (trigger)")
print("\nProposed classification:")
print("  - is_automatic_trigger: ALM > 0 (fault-initiated acquisition)")
print("  - is_manual_acquisition: ALM = 0 (operator-initiated or normal)")
print("  - loop_state: LOOP field (control system state, separate concept)")
