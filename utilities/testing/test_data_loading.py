#!/usr/bin/env python3
"""
Quick test script to verify data loading works
"""

import sys
from pathlib import Path

# Add PyPostMortem to path
pypostmortem_path = Path(__file__).parent.parent / 'data_llrf' / 'Time2Feat_test'
sys.path.insert(0, str(pypostmortem_path))

print(f"Testing PyPostMortem import from: {pypostmortem_path}")
print(f"Path exists: {pypostmortem_path.exists()}")

try:
    from PyPostMortem.PyPostMortem import Read_Signals
    print("✓ PyPostMortem imported successfully!")
except ImportError as e:
    print(f"✗ Failed to import PyPostMortem: {e}")
    sys.exit(1)

# Find a test file
DATA_DIR = Path(__file__).parent.parent / 'data_llrf' / 'data'
print(f"\nSearching for test file in: {DATA_DIR}")

test_file = None
for year_dir in sorted(DATA_DIR.iterdir()):
    if not year_dir.is_dir() or year_dir.name.startswith('.'):
        continue
    for cryo_dir in sorted(year_dir.iterdir()):
        if not cryo_dir.is_dir() or cryo_dir.name.startswith('.'):
            continue
        for cav_dir in sorted(cryo_dir.iterdir()):
            if not cav_dir.is_dir() or cav_dir.name.startswith('.'):
                continue
            for file_path in sorted(cav_dir.iterdir()):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    test_file = file_path
                    break
            if test_file:
                break
        if test_file:
            break
    if test_file:
        break

if not test_file:
    print("✗ No test file found!")
    sys.exit(1)

print(f"✓ Found test file: {test_file.name}")

# Try to load it
print("\nAttempting to load file...")
try:
    parameters, _, df_signals, _, _ = Read_Signals(
        str(test_file),
        compute_defauts=False,
        compute_etats=False,
        plot_signaux=False,
        show_header=False
    )

    print("✓ File loaded successfully!")
    print(f"\nParameters:")
    for key in ['DBNAME', 'DATE', 'NROW', 'POSTROW', 'NDEC', 'KPI', 'LOOP']:
        print(f"  {key}: {parameters.get(key, 'N/A')}")

    print(f"\nSignal data:")
    print(f"  Shape: {df_signals.shape}")
    print(f"  Columns: {df_signals.columns.tolist()}")

    print("\n" + "="*60)
    print("✓ DATA LOADING TEST PASSED!")
    print("="*60)

except Exception as e:
    print(f"✗ Error loading file: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
