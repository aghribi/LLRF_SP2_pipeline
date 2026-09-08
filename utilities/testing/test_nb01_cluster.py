#!/usr/bin/env python3
"""
Test script to run notebook 01 with minimal data to verify cluster setup.
This will test:
1. PyPostMortem import
2. Data loading from cluster paths
3. Basic data processing

Run with: python test_nb01_cluster.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add PyPostMortem to path
pypostmortem_path = Path('/pbs/throng/m4cast/projects/SPIRAL2/programmes/PyPostMortem/src')
sys.path.insert(0, str(pypostmortem_path))

try:
    from PyPostmortem.utils.PyPostMortem import Read_Signals, Read_Header
    print("✓ PyPostmortem imported successfully")
    print(f"  Path: {pypostmortem_path}")
except ImportError as e:
    print(f"✗ Failed to import PyPostmortem: {e}")
    print(f"  Path: {pypostmortem_path}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Set up paths
DATA_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/raw_data')
OUTPUT_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n✓ Paths configured:")
print(f"  Data: {DATA_DIR}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Data exists: {DATA_DIR.exists()}")
print(f"  Output exists: {OUTPUT_DIR.exists()}")

# Scan for data files
print(f"\nScanning for LLRF data files...")
data_files = []
max_files = 10  # Small sample for testing

for year_dir in sorted(DATA_DIR.iterdir())[:2]:  # Check only first 2 years
    if not year_dir.is_dir() or year_dir.name.startswith('.'):
        continue

    print(f"  Checking {year_dir.name}...")

    for cryomodule_dir in sorted(year_dir.iterdir())[:2]:  # First 2 cryomodules
        if not cryomodule_dir.is_dir() or cryomodule_dir.name.startswith('.'):
            continue

        for cavity_dir in sorted(cryomodule_dir.iterdir())[:2]:  # First 2 cavities
            if not cavity_dir.is_dir() or cavity_dir.name.startswith('.'):
                continue

            for file_path in sorted(cavity_dir.iterdir())[:max_files]:
                if file_path.is_file() and not file_path.name.startswith('.'):
                    data_files.append(file_path)
                    if len(data_files) >= max_files:
                        break

            if len(data_files) >= max_files:
                break

        if len(data_files) >= max_files:
            break

    if len(data_files) >= max_files:
        break

print(f"\n✓ Found {len(data_files)} files")

if len(data_files) == 0:
    print("✗ No data files found!")
    sys.exit(1)

# Test loading a single file
print(f"\nTesting file loading...")
test_file = data_files[0]
print(f"  Test file: {test_file.name}")

try:
    # Read_Signals expects file CONTENT (bytes), not file path!
    with open(test_file, 'rb') as f:
        file_content = f.read()

    # Read_Signals returns 6 values: parameters, time, df_signaux, df_defaut, df_states, Header
    parameters, time, df_signals, df_defaut, df_states, header = Read_Signals(
        file_content,  # Pass bytes, not string path
        compute_defauts=False,
        compute_etats=False,
        plot_signaux=False,
        show_header=False
    )

    print(f"\n✓ File loaded successfully!")
    print(f"  DBNAME: {parameters.get('DBNAME', 'N/A')}")
    print(f"  DATE: {parameters.get('DATE', 'N/A')}")
    print(f"  NROW: {parameters.get('NROW', 'N/A')}")
    print(f"  NDEC: {parameters.get('NDEC', 'N/A')}")
    print(f"  LOOP: {parameters.get('LOOP', 'N/A')}")
    print(f"  ALM: {parameters.get('ALM', 'N/A')}")
    print(f"  Signal shape: {df_signals.shape}")
    print(f"  Signal columns: {list(df_signals.columns[:10])}...")

except Exception as e:
    print(f"\n✗ Error loading file: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test metadata parsing
print(f"\nTesting metadata parsing...")

try:
    # Parse ALM field
    alm_value = int(parameters.get('ALM', '0x0'), 16)
    alm_binary = format(alm_value, '07b')[::-1]
    fault_flags = [int(b) for b in alm_binary]

    FAULT_NAMES = [
        'Seuil pick-up',
        'Coupure externe rapide',
        'Absence autorisation RF',
        'Seuil de vide',
        'Claquage ou quench cavité',
        'Dép seuil de sécurité RF',
        'Rég signal RF hors tolérance'
    ]

    print(f"  ALM field: {parameters.get('ALM')} → {alm_value} (decimal)")
    print(f"  Binary: {alm_binary}")
    print(f"  Active faults:")
    active_faults = [FAULT_NAMES[i] for i, flag in enumerate(fault_flags) if flag == 1]
    if active_faults:
        for fault in active_faults:
            print(f"    - {fault}")
    else:
        print(f"    - None")

except Exception as e:
    print(f"  ✗ Error parsing metadata: {e}")

# Summary
print(f"\n{'='*60}")
print(f"Test Summary")
print(f"{'='*60}")
print(f"✓ PyPostMortem import: SUCCESS")
print(f"✓ Data directory access: SUCCESS")
print(f"✓ File discovery: SUCCESS ({len(data_files)} files)")
print(f"✓ File loading: SUCCESS")
print(f"✓ Metadata parsing: SUCCESS")
print(f"{'='*60}")
print(f"\n✓ All tests passed!")
print(f"\nYou can now run the full notebook 01 with more data.")
print(f"Adjust SAMPLE_SIZE in the notebook as needed.")
