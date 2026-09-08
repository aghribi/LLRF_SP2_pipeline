#!/usr/bin/env python3
"""
Simple script to manually parse header and check ACQ channels
"""

import struct

# Choose a file that fails with ACQ3 error
failing_file = "/sps/m4cast/_spiral2_data/_llrf_data/raw_data/2019/CMA/CMA10/2019 11 06 CMA10_0029"

print(f"Investigating file: {failing_file}")
print("=" * 80)

try:
    # Read the file
    with open(failing_file, 'rb') as f:
        file_content = f.read()

    print(f"File size: {len(file_content)} bytes\n")

    # Read header size (first 4 bytes, big-endian integer)
    header_size = struct.unpack('>i', file_content[0:4])[0]
    print(f"Header size: {header_size} bytes")

    # Read header content (skip first 4 bytes)
    header_bytes = file_content[4:4+header_size]
    header_str = header_bytes.decode('utf-8', errors='ignore')

    # Parse header
    Header = {}
    for line in header_str.split('\n'):
        line = line.strip()
        if '=' in line:
            key, rest = line.split('=', 1)
            key = key.strip()

            # Parse value and unit
            rest = rest.strip()
            if '(' in rest and ')' in rest:
                # Format: "value (unit)"
                value = rest[:rest.rfind('(')].strip()
                unit = rest[rest.rfind('(')+1:rest.rfind(')')].strip()
            else:
                value = rest
                unit = ''

            Header[key] = {'Valeur': value, 'Unité': unit}

    print(f"Total header entries: {len(Header)}\n")

    # Check which ACQ channels are present
    print("ACQ CHANNELS (ACQ1-ACQ16):")
    print("-" * 80)

    acq_present = []
    acq_missing = []

    for i in range(1, 17):
        acq_key = f"ACQ{i}"
        if acq_key in Header:
            acq_present.append(acq_key)
            print(f"  ✅ {acq_key}: {Header[acq_key]['Valeur']}")
        else:
            acq_missing.append(acq_key)
            print(f"  ❌ {acq_key}: MISSING")

    print()
    print(f"Summary: {len(acq_present)}/16 ACQ channels present")
    print(f"Missing: {', '.join(acq_missing)}")
    print()

    # Show a few more interesting header fields
    print("OTHER HEADER FIELDS (sample):")
    print("-" * 80)
    interesting_keys = ['NDEC', 'KPI', 'ALM', 'LOOP', 'CAV', 'DATE', 'HEURE']
    for key in interesting_keys:
        if key in Header:
            print(f"  {key}: {Header[key]['Valeur']}")

    print()

    # Now try to see what happens if we try to use only existing ACQ channels
    print("WORKAROUND POSSIBILITY:")
    print("-" * 80)
    print("PyPostMortem tries to access ALL ACQ1-ACQ16, but we could:")
    print("  1. Patch PyPostMortem to only use available ACQ channels")
    print("  2. Skip files with missing ACQ channels")
    print("  3. Report this to PyPostMortem maintainer")
    print()
    print(f"For this file: Would need to skip ACQ channels: {', '.join(acq_missing)}")

except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
