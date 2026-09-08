#!/usr/bin/env python3
"""
Check if ACQ3-failing files would pass quality filters (NDEC=200)
"""

import struct
from pathlib import Path

# List of files that failed with ACQ3 error (from job output)
failing_files = [
    "/sps/m4cast/_spiral2_data/_llrf_data/raw_data/2019/CMA/CMA10/2019 11 06 CMA10_0029",
    "/sps/m4cast/_spiral2_data/_llrf_data/raw_data/2019/CMB/CMB01A/2019 11 26 CMB01-CAV1_0007",
    "/sps/m4cast/_spiral2_data/_llrf_data/raw_data/2019/CMB/CMB01A/2019 11 26 CMB01-CAV1_0009",
    "/sps/m4cast/_spiral2_data/_llrf_data/raw_data/2019/CMB/CMB02B/2019 11 26 CMB02-CAV2_0002",
    "/sps/m4cast/_spiral2_data/_llrf_data/raw_data/2020/CMA/CMA11/2020 07 30 CMA11_0002",
]

def check_file_quality(file_path):
    """Quick header parse to check NDEC and KPI"""
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()

        # Read header size
        header_size = struct.unpack('>i', file_content[0:4])[0]

        # Read header content
        header_bytes = file_content[4:4+header_size]
        header_str = header_bytes.decode('utf-8', errors='ignore')

        # Parse header
        Header = {}
        for line in header_str.split('\n'):
            line = line.strip()
            if '=' in line:
                key, rest = line.split('=', 1)
                key = key.strip()
                rest = rest.strip()

                # Simple value extraction
                if '(' in rest:
                    value = rest[:rest.rfind('(')].strip()
                else:
                    value = rest

                Header[key] = value

        # Extract quality metrics
        ndec = Header.get('NDEC', 'N/A')
        kpi = Header.get('KPI', 'N/A')
        alm = Header.get('ALM', 'N/A')
        num_acq = sum(1 for k in Header.keys() if k.startswith('ACQ') and k[3:].isdigit())

        # Check quality filters
        try:
            ndec_int = int(ndec)
            passes_ndec = (ndec_int == 200)
        except:
            passes_ndec = False

        try:
            kpi_float = float(kpi)
            passes_kpi = (kpi_float >= 10.0)
        except:
            passes_kpi = False

        return {
            'file': file_path.split('/')[-1],
            'ndec': ndec,
            'kpi': kpi,
            'alm': alm,
            'num_acq': num_acq,
            'passes_ndec': passes_ndec,
            'passes_kpi': passes_kpi,
            'would_be_used': passes_ndec and passes_kpi
        }

    except Exception as e:
        return {
            'file': file_path.split('/')[-1],
            'error': str(e)
        }

print("Checking ACQ3-failing files for quality filter compliance")
print("=" * 100)
print()

results = []
for file_path in failing_files:
    result = check_file_quality(file_path)
    results.append(result)

    if 'error' in result:
        print(f"❌ {result['file']}: Error - {result['error']}")
    else:
        status = "✅ WOULD BE USED" if result['would_be_used'] else "❌ FILTERED OUT"
        print(f"{status} | {result['file']}")
        print(f"    NDEC: {result['ndec']} {'✅' if result['passes_ndec'] else '❌ (need 200)'}")
        print(f"    KPI: {result['kpi']} {'✅' if result['passes_kpi'] else '❌ (need ≥10)'}")
        print(f"    ALM: {result['alm']}")
        print(f"    ACQ channels: {result['num_acq']}/16")
        print()

# Summary
print("=" * 100)
print("SUMMARY:")
would_be_used = sum(1 for r in results if r.get('would_be_used', False))
print(f"  Files that would pass quality filters: {would_be_used}/{len(results)}")
print(f"  Files filtered out anyway: {len(results) - would_be_used}/{len(results)}")
print()
if would_be_used == 0:
    print("✅ GOOD NEWS: All ACQ3-failing files would be filtered out by quality filters anyway!")
    print("   No need to fix ACQ3 issue - these files aren't useful for analysis.")
else:
    print("⚠️  WARNING: Some ACQ3-failing files would pass quality filters.")
    print("   Fixing ACQ3 issue could recover useful data.")
