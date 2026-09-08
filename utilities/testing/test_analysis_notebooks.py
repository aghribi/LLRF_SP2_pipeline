#!/usr/bin/env python3
"""
Test all analysis notebooks by executing them cell by cell.
Identifies errors and reports which cells fail.
"""

import json
import sys
import traceback
from pathlib import Path
from io import StringIO
import contextlib

# Analysis notebooks directory
ANALYSIS_DIR = Path(__file__).parent / 'cluster_notebooks_analysis'

# Notebooks to test (in order)
NOTEBOOKS = [
    '01_data_overview_analysis.ipynb',
    '02_signal_visualization.ipynb',
    '03_feature_exploration.ipynb',
    '04_anomaly_results.ipynb',
    '05_precursor_analysis.ipynb',
    '06_binary_classification.ipynb',
    '07_multilabel_results.ipynb',
    '08_rootcause_analysis.ipynb',
    '09_clustering_results.ipynb',
    '10_comprehensive_dashboard.ipynb',
]


class NotebookTester:
    """Test a notebook cell by cell."""

    def __init__(self, notebook_path):
        self.notebook_path = notebook_path
        self.notebook_name = notebook_path.name
        self.errors = []
        self.warnings = []
        self.cell_count = 0
        self.code_cell_count = 0
        self.successful_cells = 0
        self.failed_cells = 0

        # Execution namespace (shared across cells)
        self.namespace = {}

    def load_notebook(self):
        """Load the notebook JSON."""
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            self.notebook = json.load(f)
        self.cells = self.notebook.get('cells', [])
        self.cell_count = len(self.cells)

    def execute_cell(self, cell, cell_index):
        """Execute a single code cell."""
        source = cell.get('source', [])
        if isinstance(source, list):
            code = ''.join(source)
        else:
            code = source

        # Skip empty cells
        if not code.strip():
            return True, None

        # Skip magic commands that won't work in script mode
        if '%matplotlib' in code or '%%' in code:
            # Handle inline matplotlib config
            modified_code = code.replace('%matplotlib inline', '')
            modified_code = modified_code.replace('%matplotlib notebook', '')
            code = modified_code

        # Capture output
        output_buffer = StringIO()

        try:
            with contextlib.redirect_stdout(output_buffer):
                with contextlib.redirect_stderr(output_buffer):
                    exec(code, self.namespace)
            return True, output_buffer.getvalue()
        except Exception as e:
            error_msg = f"Cell {cell_index}: {type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()
            return False, (error_msg, tb, code)

    def test_notebook(self):
        """Test all cells in the notebook."""
        print(f"\n{'='*70}")
        print(f"Testing: {self.notebook_name}")
        print(f"{'='*70}")

        self.load_notebook()

        for idx, cell in enumerate(self.cells, 1):
            cell_type = cell.get('cell_type', 'unknown')

            if cell_type == 'markdown':
                continue

            if cell_type == 'code':
                self.code_cell_count += 1
                success, result = self.execute_cell(cell, idx)

                if success:
                    self.successful_cells += 1
                    print(f"  ✓ Cell {idx:3d} passed")
                else:
                    self.failed_cells += 1
                    error_msg, tb, code = result
                    self.errors.append({
                        'cell_index': idx,
                        'error': error_msg,
                        'traceback': tb,
                        'code': code
                    })
                    print(f"  ✗ Cell {idx:3d} FAILED: {error_msg}")

    def print_summary(self):
        """Print test summary."""
        print(f"\n{'-'*70}")
        print(f"Summary for {self.notebook_name}:")
        print(f"  Total cells: {self.cell_count}")
        print(f"  Code cells: {self.code_cell_count}")
        print(f"  Successful: {self.successful_cells}")
        print(f"  Failed: {self.failed_cells}")

        if self.errors:
            print(f"\n  ❌ ERRORS FOUND ({len(self.errors)}):")
            for err in self.errors:
                print(f"\n    Cell {err['cell_index']}:")
                print(f"    Error: {err['error']}")
                print(f"    Code preview: {err['code'][:100]}...")
        else:
            print(f"  ✅ All cells passed!")
        print(f"{'-'*70}")

        return len(self.errors) == 0


def test_all_notebooks():
    """Test all analysis notebooks."""

    print("="*70)
    print("ANALYSIS NOTEBOOKS TESTING")
    print("="*70)
    print(f"Testing directory: {ANALYSIS_DIR}")
    print(f"Number of notebooks: {len(NOTEBOOKS)}")

    results = {}
    all_errors = []

    for nb_name in NOTEBOOKS:
        nb_path = ANALYSIS_DIR / nb_name

        if not nb_path.exists():
            print(f"\n⚠️  Notebook not found: {nb_name}")
            continue

        tester = NotebookTester(nb_path)
        tester.test_notebook()
        success = tester.print_summary()

        results[nb_name] = {
            'success': success,
            'errors': len(tester.errors),
            'total_cells': tester.code_cell_count
        }

        if tester.errors:
            all_errors.extend([{
                'notebook': nb_name,
                **err
            } for err in tester.errors])

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    total_notebooks = len(results)
    passed_notebooks = sum(1 for r in results.values() if r['success'])
    failed_notebooks = total_notebooks - passed_notebooks
    total_errors = sum(r['errors'] for r in results.values())

    print(f"Notebooks tested: {total_notebooks}")
    print(f"Passed: {passed_notebooks}")
    print(f"Failed: {failed_notebooks}")
    print(f"Total errors: {total_errors}")

    if all_errors:
        print(f"\n❌ ERRORS BY NOTEBOOK:")
        for nb_name, result in results.items():
            if result['errors'] > 0:
                print(f"  {nb_name}: {result['errors']} errors")

        # Write detailed error report
        error_report_path = ANALYSIS_DIR / 'TEST_ERRORS.txt'
        with open(error_report_path, 'w') as f:
            f.write("NOTEBOOK TESTING ERROR REPORT\n")
            f.write("="*70 + "\n\n")

            for err in all_errors:
                f.write(f"Notebook: {err['notebook']}\n")
                f.write(f"Cell: {err['cell_index']}\n")
                f.write(f"Error: {err['error']}\n")
                f.write(f"Code:\n{err['code']}\n")
                f.write(f"Traceback:\n{err['traceback']}\n")
                f.write("-"*70 + "\n\n")

        print(f"\nDetailed error report written to: {error_report_path}")
    else:
        print("\n✅ All notebooks passed all tests!")

    print("="*70)

    return failed_notebooks == 0


if __name__ == '__main__':
    success = test_all_notebooks()
    sys.exit(0 if success else 1)
