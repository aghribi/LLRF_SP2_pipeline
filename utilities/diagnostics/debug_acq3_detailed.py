#!/usr/bin/env python3
"""
Detailed debug script to investigate ACQ3 error - examine Header before crash
"""

import sys
import struct
import pandas as pd
from pathlib import Path

# Add PyPostMortem to path
pypm_path = Path('/pbs/throng/m4cast/projects/SPIRAL2/programmes/PyPostMortem/src')
sys.path.insert(0, str(pypm_path))

# Import just what we need from PyPostMortem to call read_head directly
from PyPostmortem.utils.PyPostMortem import read_head

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

    # Call read_head directly (this is where the error occurs)
    print("Attempting to read header...")

    # We need to replicate what read_head does but stop before the crash
    # Let me try to import and patch it
    import PyPostmortem.utils.PyPostMortem as pm

    # Call the internal header reading without the df.columns line
    # This is a bit hacky but let's try
    try:
        Header, df = read_head(file_content)
        print("✅ Header read successfully (shouldn't get here)")
    except KeyError as e:
        print(f"❌ Expected KeyError: {e}")
        print()

        # Now let's manually parse the header to see what ACQ channels exist
        # We'll do a simplified version of what read_head does
        print("Attempting to parse Header manually...")

        # Read header size
        header_size = struct.unpack('>i', file_content[0:4])[0]
        print(f"Header size: {header_size} bytes")

        # Read header content
        header_content = file_content[4:4+header_size].decode('utf-8', errors='ignore')

        # Parse header line by line
        Header = {}
        lines = header_content.split('\n')

        for line in lines:
            if '=' in line:
                parts = line.split('=')
                if len(parts) >= 2:
                    key = parts[0].strip()
                    value_unit = parts[1].strip()

                    # Try to parse value and unit
                    if ';' in value_unit:
                        val, unit = value_unit.split(';', 1)
                        Header[key] = {'Valeur': val.strip(), 'Unité': unit.strip()}
                    else:
                        Header[key] = {'Valeur': value_unit, 'Unité': ''}

        print(f"Total header entries: {len(Header)}")
        print()

        # Check which ACQ channels are present
        print("ACQ CHANNELS PRESENT:")
        print("-" * 80)
        acq_channels = {}
        for i in range(1, 17):
            acq_key = f"ACQ{i}"
            if acq_key in Header:
                acq_channels[acq_key] = Header[acq_key]['Valeur']
                print(f"  ✅ {acq_key}: {Header[acq_key]['Valeur']}")
            else:
                print(f"  ❌ {acq_key}: MISSING")

        print()
        print(f"ACQ channels present: {len(acq_channels)}/16")
        print()

        # Print first 20 header keys to see what else is there
        print("SAMPLE HEADER KEYS (first 30):")
        print("-" * 80)
        for i, key in enumerate(sorted(Header.keys())[:30]):
            val = Header[key]
            if isinstance(val, dict):
                print(f"  {key}: {val.get('Valeur', val)}")
            else:
                print(f"  {key}: {val}")

except Exception as e:
    print(f"❌ Unexpected error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
