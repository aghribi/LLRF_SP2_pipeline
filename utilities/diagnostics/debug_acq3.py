#!/usr/bin/env python3
"""
Debug script to investigate ACQ3 error
"""

import sys
from pathlib import Path

# Add PyPostMortem to path
pypm_path = Path('/pbs/throng/m4cast/projects/SPIRAL2/programmes/PyPostMortem/src')
sys.path.insert(0, str(pypm_path))

from PyPostmortem.utils.PyPostMortem import Read_Signals

# Choose a file that fails with ACQ3 error
failing_file = "/sps/m4cast/_spiral2_data/_llrf_data/raw_data/2019/CMA/CMA10/2019 11 06 CMA10_0029"

print(f"Investigating file: {failing_file}")
print("=" * 80)

try:
    # Read the file
    with open(failing_file, 'rb') as f:
        file_content = f.read()

    print(f"File size: {len(file_content)} bytes")
    print()

    # Try to read signals
    print("Attempting to read signals...")
    parameters, time_data, df_signals, df_defaut, df_states, header = Read_Signals(
        file_content,
        compute_defauts=False,
        compute_etats=False,
        plot_signaux=False,
        show_header=False
    )

    print("✅ Successfully read file!")
    print()

    # Print parameters structure
    print("PARAMETERS (metadata):")
    print("-" * 80)
    for key in sorted(parameters.keys()):
        val = parameters[key]
        if isinstance(val, dict) and 'Valeur' in val:
            print(f"  {key}: {val['Valeur']} {val.get('Unité', '')}")
        else:
            print(f"  {key}: {val}")
    print()

    # Print signals available
    print("SIGNALS (df_signals columns):")
    print("-" * 80)
    print(f"  Total signals: {len(df_signals.columns)}")
    print(f"  Signal names: {list(df_signals.columns)}")
    print()

    # Check if ACQ3 is in signals
    if 'ACQ3' in df_signals.columns:
        print("  ✅ ACQ3 found in signals")
        print(f"     Shape: {df_signals['ACQ3'].shape}")
        print(f"     First 5 values: {df_signals['ACQ3'].head().tolist()}")
    else:
        print("  ❌ ACQ3 NOT found in signals")
    print()

    # Print DataFrame info
    print("DATAFRAME INFO:")
    print("-" * 80)
    print(df_signals.info())

except KeyError as e:
    print(f"❌ KeyError occurred: {e}")
    print()
    print("This is the ACQ3 error we're investigating!")

    # Try to see what happened before the error
    import traceback
    print()
    print("FULL TRACEBACK:")
    print("-" * 80)
    traceback.print_exc()

except Exception as e:
    print(f"❌ Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
