#!/usr/bin/env python3
"""
Step 02: Signal Preprocessing with Temporal Analysis

Description: Preprocesses LLRF post-mortem signals for feature engineering and ML.
             Applies filtering, normalization, computes derivatives, and segments
             the pre-trigger window for precursor analysis.

Input:  cooked_data/step_01_loading/metadata.csv (from Step 01)
        Raw binary files (via PyPostMortem)

Output: cooked_data/step_02_preprocessing/preprocessed_data.pkl

Processing Steps:
1. Load metadata from Step 01
2. Load raw signals with PyPostMortem
3. Temporal alignment (4000-sample window: 3000 pre + 1000 post trigger)
4. High-pass Butterworth filter (10 Hz, DC removal)
5. Z-score normalization (zero mean, unit variance)
6. Compute temporal derivatives (velocity, acceleration)
7. Segment pre-trigger window (5 segments for precursor analysis)

Usage:
    python prepare_02_preprocess.py \
        --input /path/to/step_01_loading \
        --output /path/to/step_02_preprocessing
"""

import sys
import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add PyPostMortem to path
PYPOSTMORTEM_PATH = Path('/pbs/throng/m4cast/projects/SPIRAL2/programmes/PyPostMortem/src')
sys.path.insert(0, str(PYPOSTMORTEM_PATH))

from PyPostmortem.utils.PyPostMortem import Read_Signals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step_02_preprocess.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Step02Processor:
    """Processor for Step 02: Signal Preprocessing"""

    # Signal columns to extract (8 primary channels)
    SIGNAL_COLS = [
        "A Ucr",          # Control reference amplitude
        "A Uamp",         # Amplifier voltage
        "vide",           # Vacuum level
        "courant pickup", # Pickup current
        "Ucav",           # Cavity voltage (MV/m)
        "PhaseCav",       # Cavity phase (degrees)
        "Uci",            # Input voltage (kW)
        "PhaseUci"        # Input phase (degrees)
    ]

    # Preprocessing configuration
    CONFIG = {
        'delta_pre': 3000,      # Samples before trigger
        'delta_post': 1000,     # Samples after trigger
        'cutoff_freq': 10,      # High-pass filter cutoff (Hz)
        'filter_order': 1,      # Butterworth filter order
        'target_ndec': 200,     # Target NDEC for standardization

        # Temporal segmentation boundaries (indices in pre-trigger window)
        'segments': {
            'early': (0, 600),          # T-170ms to T-136ms
            'mid_early': (600, 1200),   # T-136ms to T-102ms
            'mid': (1200, 1800),        # T-102ms to T-68ms
            'mid_late': (1800, 2400),   # T-68ms to T-34ms
            'late': (2400, 3000),       # T-34ms to T=0
        }
    }

    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.df_metadata = None
        self.preprocessed_signals = []
        self.first_derivatives = []
        self.second_derivatives = []
        self.preprocessed_metadata = []
        self.failed_events = []

    @staticmethod
    def butter_highpass(cutoff, fs, order=1):
        """
        Design high-pass Butterworth filter.

        Args:
            cutoff: Cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Filter order

        Returns:
            b, a: Numerator and denominator polynomials
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    @staticmethod
    def highpass_filter(data, cutoff, fs, order=1):
        """
        Apply high-pass filter to remove DC offset and low-frequency drift.

        Args:
            data: 1D or 2D array (samples,) or (samples, signals)
            cutoff: Cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Filter order

        Returns:
            Filtered data (same shape as input)
        """
        b, a = Step02Processor.butter_highpass(cutoff, fs, order=order)

        if data.ndim == 1:
            return filtfilt(b, a, data)
        elif data.ndim == 2:
            filtered = np.zeros_like(data)
            for i in range(data.shape[1]):
                filtered[:, i] = filtfilt(b, a, data[:, i])
            return filtered
        else:
            raise ValueError(f"Data must be 1D or 2D, got shape {data.shape}")

    @staticmethod
    def compute_derivatives(signal, dt):
        """
        Compute first and second temporal derivatives.

        Args:
            signal: Array of shape (n_samples, n_signals)
            dt: Sampling period (seconds)

        Returns:
            first_deriv: First derivative (velocity)
            second_deriv: Second derivative (acceleration)
        """
        # First derivative: forward difference
        first_deriv = np.diff(signal, axis=0) / dt
        # Pad to maintain shape
        first_deriv = np.vstack([first_deriv, first_deriv[-1:, :]])

        # Second derivative: central difference
        second_deriv = np.zeros_like(signal)
        second_deriv[1:-1, :] = (signal[2:, :] - 2*signal[1:-1, :] + signal[:-2, :]) / (dt**2)
        # Boundary conditions
        second_deriv[0, :] = second_deriv[1, :]
        second_deriv[-1, :] = second_deriv[-2, :]

        return first_deriv, second_deriv

    def preprocess_event(self, file_path):
        """
        Load and preprocess a single LLRF event.

        Returns:
            dict with processed data and metadata, or None if error
        """
        try:
            # Load signals
            with open(file_path, 'rb') as f:
                file_content = f.read()

            parameters, _, df_signals, _, _, _ = Read_Signals(
                file_content,
                compute_defauts=False,
                compute_etats=False,
                plot_signaux=False,
                show_header=False
            )

            # Extract parameters
            POSTROW = int(parameters['POSTROW'])
            NDEC = int(parameters['NDEC'])
            NROW = int(parameters['NROW'])
            cavity_type = parameters.get('CAVITE', 'CAVITE_A')

            # Compute sampling rate and period
            dt = 4 / (70.442 * NDEC) * 1e-6  # seconds per sample
            fs = 1 / dt  # Hz

            # Temporal alignment
            t0_idx = POSTROW  # Trigger index
            start_idx = max(0, t0_idx - self.CONFIG['delta_pre'])
            end_idx = min(NROW, t0_idx + self.CONFIG['delta_post'])

            df_window = df_signals.iloc[start_idx:end_idx].copy()
            actual_samples = len(df_window)

            # Extract signal columns
            signals = []
            available_cols = []
            for col in self.SIGNAL_COLS:
                if col in df_window.columns:
                    signal = df_window[col].values
                    # Replace NaN and zeros
                    signal = np.nan_to_num(signal, nan=1e-10, posinf=1e10, neginf=-1e10)
                    signal[signal == 0] = 1e-10
                    signals.append(signal)
                    available_cols.append(col)
                else:
                    # Column missing
                    signals.append(np.full(actual_samples, 1e-10))
                    available_cols.append(f"{col} (missing)")

            signal_array = np.column_stack(signals)  # (n_samples, 8)

            # Apply high-pass filter
            signal_filtered = self.highpass_filter(
                signal_array,
                self.CONFIG['cutoff_freq'],
                fs,
                order=self.CONFIG['filter_order']
            )

            # Compute temporal derivatives
            first_deriv, second_deriv = self.compute_derivatives(signal_filtered, dt)

            # Metadata
            metadata = {
                'file': str(file_path),
                'cavity_type': cavity_type,
                'fs': fs,
                'dt': dt,
                'NDEC': NDEC,
                'NROW': NROW,
                'POSTROW': POSTROW,
                'window_shape': signal_filtered.shape,
                'available_cols': available_cols,
                'start_idx': start_idx,
                'end_idx': end_idx,
            }

            return {
                'signal': signal_filtered,
                'first_derivative': first_deriv,
                'second_derivative': second_deriv,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

    @staticmethod
    def segment_pretrigger(signals, segments_config, delta_pre):
        """
        Segment the pre-trigger window.

        Args:
            signals: Array (n_events, n_samples, n_signals)
            segments_config: Dict with segment boundaries
            delta_pre: Number of pre-trigger samples

        Returns:
            dict: {segment_name: signals_segment}
        """
        segmented = {}

        for seg_name, (start, end) in segments_config.items():
            segmented[seg_name] = signals[:, start:end, :]

        # Also include post-trigger and full pre-trigger
        segmented['post_trigger'] = signals[:, delta_pre:, :]
        segmented['full_pretrigger'] = signals[:, :delta_pre, :]

        return segmented

    def load_inputs(self):
        """Load metadata from Step 01"""
        logger.info("Loading inputs from Step 01...")

        # Find metadata CSV
        metadata_files = list(self.input_dir.glob('metadata_*.csv'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata CSV found in {self.input_dir}")

        # Select most recent
        metadata_file = max(metadata_files, key=lambda p: p.stat().st_mtime)
        self.df_metadata = pd.read_csv(metadata_file)

        logger.info(f"Loaded metadata: {metadata_file.name}")
        logger.info(f"  Events: {len(self.df_metadata)}")

        # Filter by target NDEC
        df_filtered = self.df_metadata[
            self.df_metadata['NDEC'] == self.CONFIG['target_ndec']
        ].copy()

        logger.info(f"Filtered by NDEC={self.CONFIG['target_ndec']}:")
        logger.info(f"  Before: {len(self.df_metadata)} events")
        logger.info(f"  After: {len(df_filtered)} events")

        if len(df_filtered) == 0:
            logger.warning("No events with target NDEC! Using all events.")
        else:
            self.df_metadata = df_filtered

        logger.info(f"Using {len(self.df_metadata)} events for preprocessing")

    def process(self):
        """Main preprocessing logic"""
        logger.info("="*70)
        logger.info("PREPROCESSING LLRF SIGNALS")
        logger.info("="*70)

        # Process all events
        for idx in tqdm(range(len(self.df_metadata)), desc="Processing events"):
            row = self.df_metadata.iloc[idx]
            file_path = row['Fichier']

            result = self.preprocess_event(file_path)

            if result:
                self.preprocessed_signals.append(result['signal'])
                self.first_derivatives.append(result['first_derivative'])
                self.second_derivatives.append(result['second_derivative'])
                self.preprocessed_metadata.append({
                    **result['metadata'],
                    'event_id': idx,
                    'original_row': row.to_dict()
                })
            else:
                self.failed_events.append({'event_id': idx, 'file': file_path})

        logger.info("")
        logger.info("="*70)
        logger.info("Preprocessing Results:")
        logger.info("="*70)
        logger.info(f"  Total events: {len(self.df_metadata)}")
        logger.info(f"  Successful: {len(self.preprocessed_signals)}")
        logger.info(f"  Failed: {len(self.failed_events)}")
        logger.info(f"  Success rate: {100 * len(self.preprocessed_signals) / len(self.df_metadata):.1f}%")

        if self.failed_events:
            logger.warning(f"Failed events (first 5):")
            for failure in self.failed_events[:5]:
                logger.warning(f"  - Event {failure['event_id']}: {Path(failure['file']).name}")

        # Stack arrays and normalize
        logger.info("")
        logger.info("Stacking and normalizing signals...")

        all_signals = np.array(self.preprocessed_signals)
        all_first_deriv = np.array(self.first_derivatives)
        all_second_deriv = np.array(self.second_derivatives)

        logger.info(f"Stacked arrays:")
        logger.info(f"  Signals: {all_signals.shape}")
        logger.info(f"  First derivatives: {all_first_deriv.shape}")
        logger.info(f"  Second derivatives: {all_second_deriv.shape}")

        # Compute normalization statistics
        signal_means = all_signals.mean(axis=(0, 1))
        signal_stds = all_signals.std(axis=(0, 1))
        first_deriv_means = all_first_deriv.mean(axis=(0, 1))
        first_deriv_stds = all_first_deriv.std(axis=(0, 1))
        second_deriv_means = all_second_deriv.mean(axis=(0, 1))
        second_deriv_stds = all_second_deriv.std(axis=(0, 1))

        # Apply Z-score normalization
        signals_normalized = (all_signals - signal_means) / (signal_stds + 1e-10)
        first_deriv_normalized = (all_first_deriv - first_deriv_means) / (first_deriv_stds + 1e-10)
        second_deriv_normalized = (all_second_deriv - second_deriv_means) / (second_deriv_stds + 1e-10)

        logger.info("Z-score normalization applied")

        # Verify normalization
        norm_means = signals_normalized.mean(axis=(0, 1))
        norm_stds = signals_normalized.std(axis=(0, 1))
        max_mean_dev = np.abs(norm_means).max()

        logger.info(f"Normalization verification:")
        logger.info(f"  Max mean deviation: {max_mean_dev:.2e}")

        # Temporal segmentation
        logger.info("")
        logger.info("Segmenting pre-trigger window...")
        segmented_signals = self.segment_pretrigger(
            signals_normalized,
            self.CONFIG['segments'],
            self.CONFIG['delta_pre']
        )

        logger.info("Temporal segments:")
        for seg_name, seg_data in segmented_signals.items():
            logger.info(f"  {seg_name:18s}: {seg_data.shape}")

        # Prepare output data package
        preprocessed_data = {
            # Signals
            'signals_raw': all_signals,
            'signals_normalized': signals_normalized,

            # Temporal derivatives
            'first_derivative_raw': all_first_deriv,
            'first_derivative_normalized': first_deriv_normalized,
            'second_derivative_raw': all_second_deriv,
            'second_derivative_normalized': second_deriv_normalized,

            # Temporal segmentation
            'segmented_signals': segmented_signals,

            # Metadata
            'metadata': self.preprocessed_metadata,
            'signal_names': self.SIGNAL_COLS,

            # Normalization statistics
            'normalization': {
                'signal_means': signal_means,
                'signal_stds': signal_stds,
                'first_deriv_means': first_deriv_means,
                'first_deriv_stds': first_deriv_stds,
                'second_deriv_means': second_deriv_means,
                'second_deriv_stds': second_deriv_stds,
            },

            # Configuration
            'config': self.CONFIG,

            # Shape info
            'shape': all_signals.shape,
            'n_events': len(self.preprocessed_signals),
            'n_samples': all_signals.shape[1],
            'n_signals': all_signals.shape[2],
        }

        return preprocessed_data

    def save_outputs(self, results):
        """Save preprocessed data"""
        logger.info("")
        logger.info("Saving preprocessed data...")

        output_file = self.output_dir / 'preprocessed_data.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f, protocol=4)

        file_size_mb = output_file.stat().st_size / (1024**2)

        logger.info("="*70)
        logger.info("SAVED PREPROCESSED DATA")
        logger.info("="*70)
        logger.info(f"File: {output_file}")
        logger.info(f"Size: {file_size_mb:.2f} MB")
        logger.info("")
        logger.info("Contents:")
        logger.info(f"  - signals_raw: {results['signals_raw'].shape}")
        logger.info(f"  - signals_normalized: {results['signals_normalized'].shape}")
        logger.info(f"  - first_derivative_normalized: {results['first_derivative_normalized'].shape}")
        logger.info(f"  - second_derivative_normalized: {results['second_derivative_normalized'].shape}")
        logger.info(f"  - segmented_signals: {len(results['segmented_signals'])} segments")
        logger.info(f"  - metadata: {len(results['metadata'])} events")
        logger.info("")
        logger.info("✓ Data ready for Step 03 (Feature Engineering)")

    def run(self):
        """Execute full preprocessing pipeline"""
        logger.info("="*70)
        logger.info("STEP 02: SIGNAL PREPROCESSING")
        logger.info("="*70)
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("")

        self.load_inputs()
        results = self.process()
        self.save_outputs(results)

        logger.info("")
        logger.info("="*70)
        logger.info("STEP 02 COMPLETE!")
        logger.info("="*70)

        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Step 02: Signal Preprocessing')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory (step_01_loading)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory (step_02_preprocessing)')
    args = parser.parse_args()

    processor = Step02Processor(args.input, args.output)
    processor.run()


if __name__ == '__main__':
    main()
