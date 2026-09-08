#!/usr/bin/env python3
"""
LLRF Anomaly Detection - Data Preparation for Cluster
======================================================

Optimized for CC-IN2P3 SLURM cluster with multi-processing support.

Usage:
    # Single process
    python prepare_data_cluster.py --data-dir /path/to/data --output-dir /path/to/output

    # Multi-processing (recommended)
    python prepare_data_cluster.py --data-dir /path/to/data --output-dir /path/to/output --n-jobs 16

    # Process subset
    python prepare_data_cluster.py --data-dir /path/to/data --output-dir /path/to/output --max-files 1000

    # Resume from checkpoint
    python prepare_data_cluster.py --data-dir /path/to/data --output-dir /path/to/output --resume
"""

import argparse
import pickle
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class LLRFDataPreparation:
    """LLRF Data Preparation Pipeline for Cluster Processing"""

    def __init__(self, data_dir, output_dir, n_jobs=1, max_files=None, resume=False, batch_size=2000):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs
        self.max_files = max_files
        self.resume = resume
        self.batch_size = batch_size

        # Checkpoint files
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Number of jobs: {self.n_jobs}")
        logger.info(f"Max files: {self.max_files if self.max_files else 'All'}")
        logger.info(f"Resume: {self.resume}")
        logger.info(f"Batch size: {self.batch_size}")

    def collect_file_paths(self):
        """Collect all LLRF data file paths recursively"""
        logger.info("Scanning for data files recursively...")

        # Check for checkpoint
        checkpoint_file = self.checkpoint_dir / 'file_paths.pkl'
        if self.resume and checkpoint_file.exists():
            logger.info(f"Loading file paths from checkpoint: {checkpoint_file}")
            with open(checkpoint_file, 'rb') as f:
                file_paths = pickle.load(f)
            logger.info(f"Loaded {len(file_paths)} file paths from checkpoint")
            return file_paths

        # Recursively scan directory for all files
        logger.info(f"Recursively searching in: {self.data_dir}")
        all_files = list(self.data_dir.glob('**/*'))

        # Filter files
        excluded_suffixes = ['.txt', '.md', '.json', '.pyc', '.log', '.xml', '.html',
                           '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.csv']
        excluded_names = ['.directory', '.DS_Store', 'Thumbs.db']

        file_paths = []
        permission_denied_files = []
        skip_stats = {
            'not_file': 0,
            'hidden': 0,
            'excluded_name': 0,
            'excluded_suffix': 0,
            'too_small': 0,
            'permission_denied': 0,
            'other_error': 0
        }

        for f in all_files:
            # Skip if not a file
            if not f.is_file():
                skip_stats['not_file'] += 1
                continue

            # Skip hidden files
            if f.name.startswith('.'):
                skip_stats['hidden'] += 1
                continue

            # Skip excluded names
            if f.name in excluded_names:
                skip_stats['excluded_name'] += 1
                continue

            # Skip excluded suffixes
            if f.suffix.lower() in excluded_suffixes:
                skip_stats['excluded_suffix'] += 1
                continue

            # Check file size and permissions
            try:
                file_stat = f.stat()
                if file_stat.st_size < 100:  # Less than 100 bytes
                    skip_stats['too_small'] += 1
                    continue

                # Test if file is readable
                with open(f, 'rb') as test_f:
                    test_f.read(1)

            except PermissionError:
                skip_stats['permission_denied'] += 1
                permission_denied_files.append(str(f))
                continue
            except Exception as e:
                skip_stats['other_error'] += 1
                continue

            file_paths.append(f)

            if self.max_files and len(file_paths) >= self.max_files:
                break

        # Sort by path for consistency
        file_paths = sorted(file_paths)

        if self.max_files:
            file_paths = file_paths[:self.max_files]

        # Report statistics
        logger.info(f"Found {len(file_paths)} readable data files out of {len(all_files)} total items")
        logger.info("File filtering statistics:")
        logger.info(f"  - Not files (directories, etc.): {skip_stats['not_file']}")
        logger.info(f"  - Hidden files: {skip_stats['hidden']}")
        logger.info(f"  - Excluded names: {skip_stats['excluded_name']}")
        logger.info(f"  - Excluded suffixes: {skip_stats['excluded_suffix']}")
        logger.info(f"  - Too small (<100 bytes): {skip_stats['too_small']}")
        logger.info(f"  - PERMISSION DENIED: {skip_stats['permission_denied']}")
        logger.info(f"  - Other errors: {skip_stats['other_error']}")

        if skip_stats['permission_denied'] > 0:
            logger.warning("=" * 80)
            logger.warning(f"WARNING: {skip_stats['permission_denied']} files were skipped due to permission denied!")
            logger.warning("You may need to:")
            logger.warning("  1. Check if you're in the correct group: groups | grep m4cast")
            logger.warning("  2. Contact system admin to be added to the m4cast group")
            logger.warning("  3. Or ask admin to change file permissions to allow read access")
            logger.warning("=" * 80)

            # Save list of permission-denied files for debugging
            perm_denied_file = self.output_dir / 'permission_denied_files.txt'
            with open(perm_denied_file, 'w') as f:
                f.write(f"Permission Denied Files ({len(permission_denied_files)} total)\n")
                f.write("=" * 80 + "\n\n")
                for denied_file in permission_denied_files[:100]:  # Save first 100
                    f.write(f"{denied_file}\n")
                if len(permission_denied_files) > 100:
                    f.write(f"\n... and {len(permission_denied_files) - 100} more files\n")
            logger.warning(f"Sample of inaccessible files saved to: {perm_denied_file}")

        if file_paths:
            logger.info(f"Sample files found:")
            for f in file_paths[:5]:
                logger.info(f"  - {f.relative_to(self.data_dir)} ({f.stat().st_size} bytes)")

        # Save checkpoint
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(file_paths, f)
        logger.info(f"Saved file paths checkpoint: {checkpoint_file}")

        return file_paths

    def process_single_file(self, file_path):
        """Process a single LLRF data file (for multiprocessing)"""
        try:
            # Import here to avoid issues with multiprocessing
            try:
                # Add PyPostMortem src directory to path if not already there
                # Use absolute cluster path
                pypm_path = Path('/pbs/throng/m4cast/projects/SPIRAL2/programmes/PyPostMortem/src')
                pypm_path_str = str(pypm_path.resolve())

                if pypm_path_str not in sys.path:
                    sys.path.insert(0, pypm_path_str)

                from PyPostmortem.utils.PyPostMortem import Read_Signals
            except ImportError as ie:
                error_msg = (
                    f"Missing required module 'PyPostmortem': {ie}\n"
                    f"Please ensure PyPostMortem is available at: {pypm_path_str}"
                )
                logger.error(f"Error processing {file_path}: {error_msg}")
                return {'file': str(file_path), 'success': False, 'error': error_msg}

            # Read signals and parameters (metadata)
            # Read_Signals expects file content as bytes (not file path)
            with open(file_path, 'rb') as f:
                file_content = f.read()

            # Read_Signals returns 6 values: parameters, time, df_signaux, df_defaut, df_states, Header
            parameters, _, df_signals, _, _, _ = Read_Signals(
                file_content,  # Pass bytes content, not file path
                compute_defauts=False,
                compute_etats=False,
                plot_signaux=False,
                show_header=False
            )

            # Extract metadata from parameters
            # PyPostMortem returns Header with format: {"key": {"Valeur": value, "Unité": unit}}
            metadata = {}
            for key, val_dict in parameters.items():
                if isinstance(val_dict, dict) and 'Valeur' in val_dict:
                    metadata[key] = val_dict['Valeur']
                else:
                    metadata[key] = val_dict
            # Convert DataFrame columns to dict of arrays for signals
            signals = {col: df_signals[col].values for col in df_signals.columns}

            # Apply quality filters
            if not self._passes_quality_filters(metadata):
                return None

            # Extract fault labels
            fault_labels = self._extract_fault_labels(metadata)

            # Preprocess signals
            signals_processed = self._preprocess_signals(signals, metadata)

            # Engineer features
            features = self._engineer_features(signals_processed, metadata)

            return {
                'file': str(file_path),
                'metadata': metadata,
                'signals': signals_processed,
                'features': features,
                'fault_labels': fault_labels,
                'success': True
            }

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return {'file': str(file_path), 'success': False, 'error': str(e)}

    def _passes_quality_filters(self, metadata):
        """Check if event passes quality filters"""
        # Quality filters (adjust as needed)
        quality_filters = {
            'min_kpi': 10.0,
            'ndec_values': [200],
            'require_fault': False,  # Include both fault and normal events
        }

        # KPI filter (convert to float if it's a string)
        if 'KPI' in metadata:
            try:
                kpi = float(metadata['KPI'])
                if kpi < quality_filters['min_kpi']:
                    return False
            except (ValueError, TypeError):
                return False

        # NDEC filter (convert to int if it's a string)
        if 'NDEC' in metadata:
            try:
                ndec = int(metadata['NDEC'])
                if ndec not in quality_filters['ndec_values']:
                    return False
            except (ValueError, TypeError):
                return False

        return True

    def _extract_fault_labels(self, metadata):
        """Extract fault labels from metadata"""
        # Extract ALM field and decode fault flags (handle hex strings)
        alm_raw = metadata.get('ALM', 0)
        try:
            # Handle hex strings (e.g., '0x0040') or integers
            if isinstance(alm_raw, str):
                alm = int(alm_raw, 16) if alm_raw else 0
            else:
                alm = int(alm_raw) if alm_raw else 0
        except (ValueError, TypeError):
            alm = 0

        # Binary fault presence
        has_fault = alm > 0

        # Multi-label fault types (7 bits)
        fault_bits = []
        for i in range(7):
            fault_bits.append(int((alm >> i) & 1))

        return {
            'binary': int(has_fault),
            'multilabel': fault_bits,
            'alm_value': alm
        }

    def _preprocess_signals(self, signals, metadata):
        """Preprocess signals: calibration, filtering, normalization"""
        from scipy import signal as sp_signal

        processed = {}

        for signal_name, signal_data in signals.items():
            # Convert to numpy array
            data = np.array(signal_data)

            # High-pass filter (remove DC offset)
            if len(data) > 10:
                sos = sp_signal.butter(4, 0.01, 'highpass', output='sos')
                data = sp_signal.sosfilt(sos, data)

            # Normalization (z-score)
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                data = (data - mean) / std

            processed[signal_name] = data

        return processed

    def _engineer_features(self, signals, metadata):
        """Engineer features from preprocessed signals"""
        features = {}

        for signal_name, signal_data in signals.items():
            # Statistical features
            features[f'{signal_name}_mean'] = np.mean(signal_data)
            features[f'{signal_name}_std'] = np.std(signal_data)
            features[f'{signal_name}_max'] = np.max(signal_data)
            features[f'{signal_name}_min'] = np.min(signal_data)
            features[f'{signal_name}_median'] = np.median(signal_data)
            features[f'{signal_name}_rms'] = np.sqrt(np.mean(signal_data**2))

            # Temporal features
            features[f'{signal_name}_range'] = np.ptp(signal_data)

            # Spectral features (simple)
            fft = np.fft.rfft(signal_data)
            features[f'{signal_name}_peak_freq_magnitude'] = np.max(np.abs(fft))

        # Add metadata as features (convert strings to numbers if possible)
        for key, value in metadata.items():
            if isinstance(value, (int, float)):
                features[f'meta_{key}'] = value
            elif isinstance(value, str):
                # Try to convert string to number
                try:
                    # Try float first
                    features[f'meta_{key}'] = float(value)
                except ValueError:
                    # Skip non-numeric strings
                    pass

        return features

    def run_sequential(self, file_paths):
        """Run data preparation sequentially (single process)"""
        logger.info("Running sequential processing...")

        results = []
        for file_path in tqdm(file_paths, desc="Processing files"):
            result = self.process_single_file(file_path)
            if result and result['success']:
                results.append(result)

        return results

    def run_parallel(self, file_paths):
        """Run data preparation with multiprocessing and batching to avoid OOM"""
        logger.info(f"Running parallel processing with {self.n_jobs} jobs...")
        logger.info(f"Processing {len(file_paths)} files in batches of {self.batch_size}")

        # Create batch directory
        batch_dir = self.output_dir / 'batches'
        batch_dir.mkdir(exist_ok=True)

        # Process in batches
        num_batches = (len(file_paths) + self.batch_size - 1) // self.batch_size
        all_batch_files = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(file_paths))
            batch_files = file_paths[start_idx:end_idx]

            logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_files)} files)...")

            with Pool(processes=self.n_jobs) as pool:
                batch_results = list(tqdm(
                    pool.imap(self.process_single_file, batch_files),
                    total=len(batch_files),
                    desc=f"Batch {batch_idx + 1}/{num_batches}"
                ))

            # Filter successful results
            batch_results = [r for r in batch_results if r and r['success']]

            # Save batch immediately to free memory
            batch_file = batch_dir / f'batch_{batch_idx:03d}.pkl'
            with open(batch_file, 'wb') as f:
                pickle.dump({
                    'results': batch_results,
                    'n_events': len(batch_results),
                    'batch_idx': batch_idx
                }, f)

            logger.info(f"Saved batch {batch_idx + 1}: {len(batch_results)} events → {batch_file}")
            all_batch_files.append(batch_file)

            # Explicitly free memory
            del batch_results

        # Load and merge all batches
        logger.info("Merging all batches...")
        all_results = []
        for batch_file in sorted(all_batch_files):
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
                all_results.extend(batch_data['results'])
                logger.info(f"Loaded {batch_data['n_events']} events from {batch_file.name}")

        logger.info(f"Total events from all batches: {len(all_results)}")
        return all_results

    def save_results(self, results):
        """Save processed results to disk"""
        logger.info(f"Saving {len(results)} processed events...")

        # Check if results are empty
        if len(results) == 0:
            logger.error("No results to save - all files failed to process")
            raise ValueError("No data was successfully processed. Check the errors above for details.")

        # Separate components
        metadata_list = [r['metadata'] for r in results]
        features_list = [r['features'] for r in results]
        fault_labels = [r['fault_labels'] for r in results]

        # Create feature dataframe
        df_features = pd.DataFrame(features_list)

        # Create labels
        y_binary = np.array([f['binary'] for f in fault_labels])
        y_multilabel = np.array([f['multilabel'] for f in fault_labels])

        # Save preprocessed data
        preprocessed_file = self.output_dir / 'preprocessed_data.pkl'
        with open(preprocessed_file, 'wb') as f:
            pickle.dump({
                'signals_normalized': [r['signals'] for r in results],
                'metadata': metadata_list,
                'signal_names': list(results[0]['signals'].keys()) if results else [],
            }, f)
        logger.info(f"Saved preprocessed data: {preprocessed_file}")

        # Save features
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_features.fillna(0))

        # PCA (only if we have enough samples)
        # n_components must be <= min(n_samples, n_features)
        n_samples, n_features = X_scaled.shape
        max_components = min(50, n_samples, n_features)

        if max_components > 0:
            pca = PCA(n_components=max_components)
            X_pca = pca.fit_transform(X_scaled)
        else:
            pca = None
            X_pca = X_scaled  # Use scaled features if PCA not possible

        features_file = self.output_dir / 'features_engineered.pkl'
        with open(features_file, 'wb') as f:
            pickle.dump({
                'features_all': df_features,
                'feature_cols': df_features.columns.tolist(),
                'X_scaled': X_scaled,
                'X_pca': X_pca,
                'y_binary': y_binary,
                'y_multilabel': y_multilabel,
                'fault_column_names': [f'Fault_{i}' for i in range(7)],
                'scaler': scaler,
                'pca': pca,
                'metadata': metadata_list,
            }, f)
        logger.info(f"Saved features: {features_file}")

        # Save summary
        summary_file = self.output_dir / 'processing_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"LLRF Data Preparation Summary\n")
            f.write(f"=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Total events processed: {len(results)}\n")
            f.write(f"Normal events: {np.sum(y_binary == 0)}\n")
            f.write(f"Fault events: {np.sum(y_binary == 1)}\n")
            f.write(f"Features extracted: {X_scaled.shape[1]}\n")
            f.write(f"PCA components: {X_pca.shape[1]}\n")
            f.write(f"\nFiles:\n")
            f.write(f"  - preprocessed_data.pkl\n")
            f.write(f"  - features_engineered.pkl\n")
        logger.info(f"Saved summary: {summary_file}")

        return {
            'n_events': len(results),
            'n_normal': np.sum(y_binary == 0),
            'n_fault': np.sum(y_binary == 1),
            'n_features': X_scaled.shape[1]
        }

    def run(self):
        """Main execution pipeline"""
        start_time = time.time()

        # Monitor resources
        process = psutil.Process()
        logger.info(f"Initial memory usage: {process.memory_info().rss / 1e9:.2f} GB")

        # Step 1: Collect file paths
        file_paths = self.collect_file_paths()

        # Step 2: Process files
        if self.n_jobs > 1:
            results = self.run_parallel(file_paths)
        else:
            results = self.run_sequential(file_paths)

        logger.info(f"Successfully processed {len(results)}/{len(file_paths)} files")

        # Check if any files were successfully processed
        if len(results) == 0:
            logger.error("=" * 80)
            logger.error("FATAL: No files were successfully processed!")
            logger.error("=" * 80)
            logger.error("Common causes:")
            logger.error("  1. Missing or incorrectly configured PyPostmortem module")
            logger.error("     Check path: /pbs/home/a/aghribi/throng_m4cast/projects/SPIRAL2/programmes/PyPostMortem/src")
            logger.error("  2. Corrupted or incompatible data files")
            logger.error("  3. Permission issues reading files")
            logger.error("  4. No valid PostMortem data files in the specified directory")
            logger.error("=" * 80)
            logger.error("Please check the error messages above for specific failures.")
            raise RuntimeError("No data was successfully processed. See errors above.")

        # Step 3: Save results
        summary = self.save_results(results)

        # Final stats
        elapsed = time.time() - start_time
        logger.info(f"Memory usage: {process.memory_info().rss / 1e9:.2f} GB")
        logger.info(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        logger.info(f"Processing rate: {len(results)/elapsed:.2f} files/second")
        logger.info(f"\nSummary:")
        logger.info(f"  Events: {summary['n_events']}")
        logger.info(f"  Normal: {summary['n_normal']}")
        logger.info(f"  Faults: {summary['n_fault']}")
        logger.info(f"  Features: {summary['n_features']}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description='LLRF Data Preparation for Cluster',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to LLRF data directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs (default: 1, use -1 for all CPUs)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--batch-size', type=int, default=2000,
                        help='Batch size for processing (to avoid OOM, default: 2000)')

    args = parser.parse_args()

    # Determine number of jobs
    if args.n_jobs == -1:
        n_jobs = cpu_count()
        logger.info(f"Using all available CPUs: {n_jobs}")
    else:
        n_jobs = args.n_jobs

    # Run pipeline
    pipeline = LLRFDataPreparation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_jobs=n_jobs,
        max_files=args.max_files,
        resume=args.resume,
        batch_size=args.batch_size
    )

    try:
        summary = pipeline.run()
        logger.info("✓ Data preparation completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"✗ Data preparation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
