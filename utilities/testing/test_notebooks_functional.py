#!/usr/bin/env python3
"""
Test if analysis notebooks can load their data
Tests notebooks 04-10 for functionality
"""

import pickle
import numpy as np
from pathlib import Path
import sys

OUTPUT_DIR = Path('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data')

print("=" * 80)
print("ANALYSIS NOTEBOOKS FUNCTIONALITY TEST")
print("=" * 80)

# Test configuration for each notebook
notebook_tests = {
    "04_anomaly_results.ipynb": {
        "data_files": [
            OUTPUT_DIR / 'features_engineered.pkl',
            OUTPUT_DIR / 'step_04_anomaly/anomaly_baseline.pkl',
        ],
        "description": "Anomaly Detection Results"
    },
    "05_precursor_analysis.ipynb": {
        "data_files": [
            OUTPUT_DIR / 'step_05_phase0/precursor_detection.pkl',
        ],
        "description": "Precursor Detection Analysis"
    },
    "06_binary_classification.ipynb": {
        "data_files": [
            OUTPUT_DIR / 'step_06_phase1/binary_classification.pkl',
        ],
        "description": "Binary Classification Results"
    },
    "07_multilabel_results.ipynb": {
        "data_files": [
            OUTPUT_DIR / 'step_07_phase2/multilabel_triggers.pkl',
        ],
        "description": "Multi-Label Classification Results"
    },
    "08_rootcause_analysis.ipynb": {
        "data_files": [
            OUTPUT_DIR / 'step_08_phase3a/root_cause.pkl',
        ],
        "description": "Root Cause Analysis Results"
    },
    "09_clustering_results.ipynb": {
        "data_files": [
            OUTPUT_DIR / 'step_09_phase3b/subtype_clustering.pkl',
        ],
        "description": "Clustering Results"
    },
    "10_comprehensive_dashboard.ipynb": {
        "data_files": [
            OUTPUT_DIR / 'step_10_comprehensive/comprehensive_analysis.pkl',
            OUTPUT_DIR / 'step_10_comprehensive/pipeline_report.txt',
        ],
        "description": "Comprehensive Dashboard"
    },
}

all_passed = True
results = []

for notebook_name, config in notebook_tests.items():
    print(f"\n{'=' * 80}")
    print(f"Testing: {notebook_name}")
    print(f"Description: {config['description']}")
    print(f"{'=' * 80}")

    notebook_ok = True
    files_status = []

    for data_file in config['data_files']:
        try:
            if data_file.suffix == '.pkl':
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)

                if isinstance(data, dict):
                    info = f"Dict with {len(data)} keys"
                elif isinstance(data, np.ndarray):
                    info = f"Array shape {data.shape}"
                else:
                    info = f"Type: {type(data).__name__}"

                print(f"  ✓ {data_file.name}")
                print(f"    {info}")
                files_status.append(True)

            elif data_file.suffix == '.txt':
                with open(data_file, 'r') as f:
                    lines = f.readlines()
                print(f"  ✓ {data_file.name}")
                print(f"    {len(lines)} lines")
                files_status.append(True)

            else:
                print(f"  ✓ {data_file.name} (exists)")
                files_status.append(True)

        except FileNotFoundError:
            print(f"  ✗ {data_file.name} - NOT FOUND")
            print(f"    Path: {data_file}")
            notebook_ok = False
            files_status.append(False)

        except Exception as e:
            print(f"  ✗ {data_file.name} - ERROR")
            print(f"    {str(e)}")
            notebook_ok = False
            files_status.append(False)

    # Overall notebook status
    if notebook_ok and all(files_status):
        print(f"\n  ✅ {notebook_name} - FUNCTIONAL")
        results.append((notebook_name, "FUNCTIONAL"))
    else:
        print(f"\n  ❌ {notebook_name} - ISSUES FOUND")
        results.append((notebook_name, "ISSUES"))
        all_passed = False

# Summary
print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"{'=' * 80}")

functional = sum(1 for _, status in results if status == "FUNCTIONAL")
total = len(results)

print(f"\nNotebooks tested: {total}")
print(f"Functional: {functional}")
print(f"With issues: {total - functional}")

print(f"\nDetailed Results:")
for notebook, status in results:
    symbol = "✅" if status == "FUNCTIONAL" else "❌"
    print(f"  {symbol} {notebook:35} - {status}")

if all_passed:
    print(f"\n{'=' * 80}")
    print("✅ ALL NOTEBOOKS ARE FUNCTIONAL!")
    print(f"{'=' * 80}")
    print("\nAll analysis notebooks can load their required data files.")
    print("Ready for visualization and analysis.")
    sys.exit(0)
else:
    print(f"\n{'=' * 80}")
    print("⚠️  SOME NOTEBOOKS HAVE ISSUES")
    print(f"{'=' * 80}")
    print("\nSome notebooks may need updates to load pipeline outputs.")
    sys.exit(1)
