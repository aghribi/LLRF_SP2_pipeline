#!/usr/bin/env python3
"""
Diagnostic script to understand LLRF raw data characteristics
- What NDEC values exist?
- What trigger types exist (automatic vs manual)?
- What ALM values exist (fault vs no-fault)?
"""

import sys
from pathlib import Path
from collections import Counter

# Add PyPostMortem to path
pypm_path = Path('/pbs/throng/m4cast/projects/SPIRAL2/programmes/PyPostMortem/src')
sys.path.insert(0, str(pypm_path.resolve()))

from PyPostmortem.utils.PyPostMortem import Read_Signals

def diagnose_file(file_path):
    """Extract key metadata from a file"""
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()

        parameters, _, _, _, _, _ = Read_Signals(
            file_content,
            compute_defauts=False,
            compute_etats=False,
            plot_signaux=False,
            show_header=False
        )

        # Extract key parameters
        metadata = {}
        for key, val_dict in parameters.items():
            if isinstance(val_dict, dict) and 'Valeur' in val_dict:
                metadata[key] = val_dict['Valeur']
            else:
                metadata[key] = val_dict

        # Extract critical info
        ndec = metadata.get('NDEC', 'N/A')
        kpi = metadata.get('KPI', 'N/A')
        alm_raw = metadata.get('ALM', '0')

        # Parse ALM
        try:
            if isinstance(alm_raw, str):
                alm = int(alm_raw, 16) if alm_raw else 0
            else:
                alm = int(alm_raw) if alm_raw else 0
        except:
            alm = 0

        has_fault = alm > 0
        trigger_type = 'automatic' if has_fault else 'manual'

        return {
            'file': file_path.name,
            'ndec': ndec,
            'kpi': kpi,
            'alm': alm,
            'has_fault': has_fault,
            'trigger_type': trigger_type,
            'success': True
        }
    except Exception as e:
        return {
            'file': file_path.name,
            'success': False,
            'error': str(e)
        }

def main():
    data_dir = Path('/sps/m4cast/_spiral2_data/_llrf_data/raw_data')

    # Sample files (first 500) - files have NO extension, they're just files
    all_files = []
    for f in data_dir.rglob('*'):
        if f.is_file() and not f.name.startswith('.'):
            all_files.append(f)

    all_files = sorted(all_files)
    sample_files = all_files[:500]

    print(f"Total files found: {len(all_files)}")
    print(f"Analyzing sample of: {len(sample_files)} files")
    print("="*80)

    # Collect statistics
    results = []
    ndec_counter = Counter()
    kpi_values = []
    alm_counter = Counter()
    trigger_types = Counter()
    errors = []

    for file_path in sample_files:
        result = diagnose_file(file_path)

        if result['success']:
            results.append(result)
            ndec_counter[result['ndec']] += 1
            trigger_types[result['trigger_type']] += 1
            alm_counter[result['alm']] += 1

            if result['kpi'] != 'N/A':
                try:
                    kpi_values.append(float(result['kpi']))
                except:
                    pass
        else:
            errors.append(result)

    # Print statistics
    print(f"\n✅ Successfully analyzed: {len(results)}/{len(sample_files)} files")
    print(f"❌ Errors: {len(errors)}/{len(sample_files)} files")

    print(f"\n{'='*80}")
    print("NDEC VALUES DISTRIBUTION:")
    print(f"{'='*80}")
    for ndec, count in sorted(ndec_counter.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(results)
        print(f"  NDEC={ndec:>4s}: {count:>4d} files ({pct:>5.1f}%)")

    print(f"\n{'='*80}")
    print("TRIGGER TYPE DISTRIBUTION:")
    print(f"{'='*80}")
    for trigger_type, count in sorted(trigger_types.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(results)
        print(f"  {trigger_type:>10s}: {count:>4d} files ({pct:>5.1f}%)")

    print(f"\n{'='*80}")
    print("ALM VALUES DISTRIBUTION (top 10):")
    print(f"{'='*80}")
    for alm, count in sorted(alm_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = 100 * count / len(results)
        fault_str = "FAULT" if alm > 0 else "NORMAL"
        print(f"  ALM={alm:>4d} ({fault_str:>6s}): {count:>4d} files ({pct:>5.1f}%)")

    if kpi_values:
        import numpy as np
        print(f"\n{'='*80}")
        print("KPI VALUES STATISTICS:")
        print(f"{'='*80}")
        print(f"  Min:    {np.min(kpi_values):.2f}")
        print(f"  Max:    {np.max(kpi_values):.2f}")
        print(f"  Mean:   {np.mean(kpi_values):.2f}")
        print(f"  Median: {np.median(kpi_values):.2f}")

    # Critical findings
    print(f"\n{'='*80}")
    print("CRITICAL FINDINGS:")
    print(f"{'='*80}")

    # Check current filter (NDEC == 200)
    ndec_200_count = ndec_counter.get('200', 0) + ndec_counter.get(200, 0)
    ndec_200_pct = 100 * ndec_200_count / len(results) if results else 0
    print(f"  Current filter (NDEC=200): {ndec_200_count} files ({ndec_200_pct:.1f}%)")
    print(f"  Would reject: {len(results) - ndec_200_count} files ({100-ndec_200_pct:.1f}%)")

    # Check manual vs automatic
    manual_count = trigger_types.get('manual', 0)
    automatic_count = trigger_types.get('automatic', 0)
    print(f"\n  Manual acquisitions (no fault): {manual_count} files")
    print(f"  Automatic triggers (fault):     {automatic_count} files")

    if manual_count == 0:
        print(f"\n  ⚠️  WARNING: NO MANUAL ACQUISITIONS FOUND IN SAMPLE!")
        print(f"  This means NO normal (non-fault) events for ML training!")

    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print(f"{'='*80}")

    if ndec_200_pct < 50:
        print(f"  ❌ NDEC filter too restrictive!")
        print(f"     Current: NDEC == 200 only")
        print(f"     Rejects {100-ndec_200_pct:.1f}% of data")
        print(f"     Recommend: Accept all NDEC values or expand to common values")

    if manual_count == 0:
        print(f"  ❌ No manual acquisitions detected!")
        print(f"     ML training REQUIRES both fault and no-fault events")
        print(f"     Check if manual acquisitions exist in other files")

    # Sample files for inspection
    print(f"\n{'='*80}")
    print("SAMPLE FILES FOR INSPECTION:")
    print(f"{'='*80}")
    for i, result in enumerate(results[:10]):
        print(f"{i+1}. {result['file']}")
        print(f"   NDEC={result['ndec']}, KPI={result['kpi']}, ALM={result['alm']}, Type={result['trigger_type']}")

if __name__ == '__main__':
    main()
