#!/usr/bin/env python3
"""
Diagnostic script to understand Ucav/Uci filtering effectiveness
"""
import sys
import pickle
from pathlib import Path
import numpy as np

# Add PyPostMortem to path
pypm_path = Path('/pbs/throng/m4cast/projects/SPIRAL2/programmes/PyPostMortem/src')
sys.path.insert(0, str(pypm_path))

from PyPostmortem.utils.PyPostMortem import Read_Signals

def analyze_sample_files(data_dir, n_samples=100):
    """Analyze raw files to see Ucav/Uci distribution"""
    data_dir = Path(data_dir)

    # Get sample files
    all_files = sorted(list(data_dir.glob('**/*')))
    # Filter to actual data files
    excluded_suffixes = ['.txt', '.md', '.json', '.pyc', '.log', '.xml', '.html',
                       '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.csv', '.fig']
    files = [f for f in all_files if f.is_file() and f.suffix.lower() not in excluded_suffixes
             and not f.name.startswith('.') and f.stat().st_size > 100]

    print(f"Found {len(files)} total files")
    print(f"Analyzing {min(n_samples, len(files))} sample files...\n")

    ucav_values = []
    uci_values = []
    kpi_values = []

    pass_count = 0
    fail_kpi = 0
    fail_ucav = 0
    fail_uci = 0
    fail_parse = 0

    for i, file_path in enumerate(files[:n_samples]):
        try:
            # Read file
            with open(file_path, 'rb') as f:
                file_content = f.read()

            parameters, _, df_signals, _, _, _ = Read_Signals(
                file_content,
                compute_defauts=False,
                compute_etats=False,
                plot_signaux=False,
                show_header=False
            )

            # Extract metadata
            metadata = {}
            for key, val_dict in parameters.items():
                if isinstance(val_dict, dict) and 'Valeur' in val_dict:
                    metadata[key] = val_dict['Valeur']
                else:
                    metadata[key] = val_dict

            # Get signals
            signals = {col: df_signals[col].values for col in df_signals.columns}

            # Check KPI
            kpi = float(metadata.get('KPI', 0))
            kpi_values.append(kpi)

            # Check Ucav
            if 'Ucav' in signals:
                ucav = np.array(signals['Ucav'])
                ucav_mean = np.mean(np.abs(ucav))
                ucav_values.append(ucav_mean)
            else:
                ucav_mean = 0

            # Check Uci
            if 'Uci' in signals:
                uci = np.array(signals['Uci'])
                uci_mean = np.mean(np.abs(uci))
                uci_values.append(uci_mean)
            else:
                uci_mean = 0

            # Check filters
            passes = True
            if kpi < 10.0:
                fail_kpi += 1
                passes = False
            if ucav_mean < 0.1:
                fail_ucav += 1
                passes = False
            if uci_mean < 0.02:
                fail_uci += 1
                passes = False

            if passes:
                pass_count += 1

            if i < 10:
                status = "PASS" if passes else "FAIL"
                print(f"File {i+1}: {status}")
                print(f"  KPI={kpi:.2f}, Ucav_mean={ucav_mean:.4f}, Uci_mean={uci_mean:.6f}")

        except Exception as e:
            fail_parse += 1
            if i < 10:
                print(f"File {i+1}: PARSE ERROR - {str(e)[:50]}")

    print(f"\n{'='*80}")
    print("FILTERING STATISTICS (sample of {})".format(n_samples))
    print(f"{'='*80}")
    print(f"Parsing failures: {fail_parse} ({100*fail_parse/n_samples:.1f}%)")
    print(f"Failed KPI filter (< 10.0): {fail_kpi} ({100*fail_kpi/n_samples:.1f}%)")
    print(f"Failed Ucav filter (< 0.1): {fail_ucav} ({100*fail_ucav/n_samples:.1f}%)")
    print(f"Failed Uci filter (< 0.02): {fail_uci} ({100*fail_uci/n_samples:.1f}%)")
    print(f"Passed all filters: {pass_count} ({100*pass_count/n_samples:.1f}%)")

    if ucav_values:
        print(f"\nUcav distribution (MV/m):")
        print(f"  Min: {np.min(ucav_values):.4f}")
        print(f"  10th percentile: {np.percentile(ucav_values, 10):.4f}")
        print(f"  Median: {np.median(ucav_values):.4f}")
        print(f"  90th percentile: {np.percentile(ucav_values, 90):.4f}")
        print(f"  Max: {np.max(ucav_values):.4f}")
        print(f"  % above 0.1 threshold: {100*np.sum(np.array(ucav_values) > 0.1)/len(ucav_values):.1f}%")

    if uci_values:
        print(f"\nUci distribution:")
        print(f"  Min: {np.min(uci_values):.6f}")
        print(f"  10th percentile: {np.percentile(uci_values, 10):.6f}")
        print(f"  Median: {np.median(uci_values):.6f}")
        print(f"  90th percentile: {np.percentile(uci_values, 90):.6f}")
        print(f"  Max: {np.max(uci_values):.6f}")
        print(f"  % above 0.02 threshold: {100*np.sum(np.array(uci_values) > 0.02)/len(uci_values):.1f}%")

    if kpi_values:
        print(f"\nKPI distribution:")
        print(f"  Min: {np.min(kpi_values):.2f}")
        print(f"  10th percentile: {np.percentile(kpi_values, 10):.2f}")
        print(f"  Median: {np.median(kpi_values):.2f}")
        print(f"  90th percentile: {np.percentile(kpi_values, 90):.2f}")
        print(f"  Max: {np.max(kpi_values):.2f}")
        print(f"  % above 10.0 threshold: {100*np.sum(np.array(kpi_values) > 10.0)/len(kpi_values):.1f}%")

if __name__ == '__main__':
    data_dir = '/sps/m4cast/_spiral2_data/_llrf_data/raw_data'
    analyze_sample_files(data_dir, n_samples=200)
