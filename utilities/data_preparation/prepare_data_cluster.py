#!/usr/bin/env python3
"""
LLRF Anomaly Detection - Data Preparation for Cluster (IMPROVED WITH TIMEOUTS)
================================================================================

Optimized for CC-IN2P3 SLURM cluster with:
- Per-file timeout protection (prevents infinite hangs)
- Multi-processing support
- Checkpoint/resume capability
- Better error logging

FIXES:
- Added per-file timeout (default 30s) to prevent hanging on corrupted files
- Better error logging with problematic file tracking
- Resume from last successful batch

Usage:
    # Single process
    python prepare_data_cluster.py --data-dir /path/to/data --output-dir /path/to/output

    # Multi-processing (recommended)
    python prepare_data_cluster.py --data-dir /path/to/data --output-dir /path/to/output --n-jobs 16

    # Resume from last batch (after job was killed)
    python prepare_data_cluster.py --data-dir /path/to/data --output-dir /path/to/output --resume

    # Custom timeout for slow files
    python prepare_data_cluster.py --data-dir /path/to/data --output-dir /path/to/output --file-timeout 60
"""

import argparse
import pickle
import logging
import sys
import time
import signal
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count, TimeoutError as MP_TimeoutError
from functools import partial
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


class TimeoutException(Exception):
    """Exception raised when file processing times out"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("File processing timeout")


def process_single_file_with_timeout(file_path, file_timeout=30):
    """
    Process a single LLRF data file with timeout protection

    Args:
        file_path: Path to file
        file_timeout: Timeout in seconds (default 30s)

    Returns:
        dict with processing results or error info
    """
    # Set up timeout (UNIX only)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(file_timeout)

    try:
        # Import here to avoid issues with multiprocessing
        try:
            # Add PyPostMortem src directory to path if not already there
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
            return {'file': str(file_path), 'success': False, 'error': error_msg}

        # Read signals and parameters (metadata)
        with open(file_path, 'rb') as f:
            file_content = f.read()

        # Read_Signals returns 6 values: parameters, time, df_signaux, df_defaut, df_states, Header
        parameters, _, df_signals, _, _, _ = Read_Signals(
            file_content,
            compute_defauts=False,
            compute_etats=False,
            plot_signaux=False,
            show_header=False
        )

        # Cancel timeout
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

        # Extract metadata from parameters
        metadata = {}
        for key, val_dict in parameters.items():
            if isinstance(val_dict, dict) and 'Valeur' in val_dict:
                metadata[key] = val_dict['Valeur']
            else:
                metadata[key] = val_dict

        # Convert DataFrame columns to dict of arrays for signals
        signals = {col: df_signals[col].values for col in df_signals.columns}

        # Apply quality filters
        if not passes_quality_filters(metadata):
            return None

        # Extract fault labels
        fault_labels = extract_fault_labels(metadata)

        # Preprocess signals
        signals_processed = preprocess_signals(signals, metadata)

        # Engineer features
        features = engineer_features(signals_processed, metadata)

        return {
            'file': str(file_path),
            'metadata': metadata,
            'signals': signals_processed,
            'features': features,
            'fault_labels': fault_labels,
            'success': True
        }

    except TimeoutException:
        # Cancel timeout
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        error_msg = f"TIMEOUT after {file_timeout}s"
        logger.error(f"Error processing {file_path}: {error_msg}")
        return {'file': str(file_path), 'success': False, 'error': error_msg}

    except Exception as e:
        # Cancel timeout
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        logger.error(f"Error processing {file_path}: {e}")
        return {'file': str(file_path), 'success': False, 'error': str(e)}


def passes_quality_filters(metadata):
    """
    Check if event passes quality filters

    IMPORTANT: We need BOTH fault and no-fault events for ML training!
    - Fault events: ALM > 0 (automatic triggers)
    - Normal events: ALM = 0 (manual acquisitions, no faults)

    CRITICAL FIX: Removed NDEC filter that was rejecting most data!
    Previously only accepted NDEC=200, which filtered out ~60-80% of files.
    Now accepts ALL NDEC values to maximize training data.
    """
    quality_filters = {
        'min_kpi': 10.0,
        'require_fault': False,  # ✅ Include both fault and normal events
    }

    # KPI filter (minimum quality threshold)
    if 'KPI' in metadata:
        try:
            kpi = float(metadata['KPI'])
            if kpi < quality_filters['min_kpi']:
                return False
        except (ValueError, TypeError):
            return False

    # NDEC filter - REMOVED to accept all NDEC values
    # Previously this was rejecting most files by only accepting NDEC=200:
    #
    # if 'NDEC' in metadata:
    #     try:
    #         ndec = int(metadata['NDEC'])
    #         if ndec not in [200]:  # ← This rejected ~60-80% of data!
    #             return False
    #     except (ValueError, TypeError):
    #         return False
    #
    # Now we accept ALL NDEC values to maximize training data.
    # If specific NDEC filtering is needed, it can be added later.

    return True


def extract_fault_labels(metadata):
    """Extract fault labels from metadata"""
    alm_raw = metadata.get('ALM', 0)
    try:
        if isinstance(alm_raw, str):
            alm = int(alm_raw, 16) if alm_raw else 0
        else:
            alm = int(alm_raw) if alm_raw else 0
    except (ValueError, TypeError):
        alm = 0

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


def preprocess_signals(signals, metadata):
    """
    Preprocess signals: filtering, normalization, derivatives, segmentation

    Returns: dict with keys:
        - 'signals': normalized signals
        - 'first_deriv': 1st derivatives (velocity)
        - 'second_deriv': 2nd derivatives (acceleration)
        - 'segments': 5 temporal segments of pre-trigger window
    """
    from scipy import signal as sp_signal

    processed = {
        'signals': {},
        'first_deriv': {},
        'second_deriv': {},
        'segments': {}
    }

    # Configuration for temporal windows
    # Assume 4000 samples total: 3000 pre-trigger + 1000 post-trigger
    delta_pre = 3000  # Pre-trigger samples
    segment_boundaries = {
        'early': (0, 600),          # T-170ms to T-136ms
        'mid_early': (600, 1200),   # T-136ms to T-102ms
        'mid': (1200, 1800),        # T-102ms to T-68ms
        'mid_late': (1800, 2400),   # T-68ms to T-34ms
        'late': (2400, 3000),       # T-34ms to T=0
    }

    for signal_name, signal_data in signals.items():
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

        processed['signals'][signal_name] = data

        # Compute derivatives
        first_deriv = np.gradient(data)
        second_deriv = np.gradient(first_deriv)

        processed['first_deriv'][signal_name] = first_deriv
        processed['second_deriv'][signal_name] = second_deriv

        # Extract temporal segments (pre-trigger only for precursor analysis)
        if len(data) >= delta_pre:
            pre_trigger_data = data[:delta_pre]
            processed['segments'][signal_name] = {}

            for segment_name, (start, end) in segment_boundaries.items():
                if end <= len(pre_trigger_data):
                    processed['segments'][signal_name][segment_name] = pre_trigger_data[start:end]

    return processed


def engineer_features(signals_processed, metadata):
    """
    Engineer comprehensive features from preprocessed signals

    Extracts 1200+ features (ALL signals):
    - Statistical features (17 per signal × 27 signals = 459)
      Includes: mean, std, min, max, range, median, q1, q3, iqr,
               skewness, kurtosis, rms, crest_factor, peak_count,
               energy, n_zero_crossings, autocorr_lag1
    - Physics features (23 total): Q_L, detuning, control error, settling time,
                                   overshoot, phase stability, phase drift, FFT, etc.
    - Precursor features (~729): trends, CUSUM, derivatives, segments, variance ratios
    - Metadata features (~10)

    Args:
        signals_processed: dict with keys 'signals', 'first_deriv', 'second_deriv', 'segments'
        metadata: dict with ALM, LOOP, KPI, etc.

    Returns:
        dict of features
    """
    from scipy import stats as sp_stats
    from scipy.signal import find_peaks

    features = {}

    # Extract components
    signals = signals_processed['signals']
    first_deriv = signals_processed['first_deriv']
    second_deriv = signals_processed['second_deriv']
    segments = signals_processed['segments']

    # ========================================================================
    # 1. STATISTICAL FEATURES (17 per signal × 27 signals = 459 features)
    # ========================================================================
    for signal_name, signal_data in signals.items():
        # Basic statistics
        features[f'{signal_name}_mean'] = np.mean(signal_data)
        features[f'{signal_name}_std'] = np.std(signal_data)
        features[f'{signal_name}_min'] = np.min(signal_data)
        features[f'{signal_name}_max'] = np.max(signal_data)
        features[f'{signal_name}_range'] = np.ptp(signal_data)
        features[f'{signal_name}_median'] = np.median(signal_data)

        # Quartiles
        features[f'{signal_name}_q1'] = np.percentile(signal_data, 25)
        features[f'{signal_name}_q3'] = np.percentile(signal_data, 75)
        features[f'{signal_name}_iqr'] = features[f'{signal_name}_q3'] - features[f'{signal_name}_q1']

        # Distribution shape
        features[f'{signal_name}_skewness'] = sp_stats.skew(signal_data)
        features[f'{signal_name}_kurtosis'] = sp_stats.kurtosis(signal_data)

        # RMS and crest factor
        features[f'{signal_name}_rms'] = np.sqrt(np.mean(signal_data**2))
        crest = features[f'{signal_name}_max'] / features[f'{signal_name}_rms'] if features[f'{signal_name}_rms'] > 0 else 0
        features[f'{signal_name}_crest_factor'] = crest

        # Peak analysis
        peaks, properties = find_peaks(signal_data, prominence=0.5)
        features[f'{signal_name}_peak_count'] = len(peaks)

        # Energy (sum of squared values)
        features[f'{signal_name}_energy'] = np.sum(signal_data**2)

        # Zero crossings (sign changes)
        zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
        features[f'{signal_name}_n_zero_crossings'] = len(zero_crossings)

        # Autocorrelation at lag 1
        if len(signal_data) > 1:
            features[f'{signal_name}_autocorr_lag1'] = np.corrcoef(signal_data[:-1], signal_data[1:])[0, 1]
        else:
            features[f'{signal_name}_autocorr_lag1'] = 0.0

    # ========================================================================
    # 2. PHYSICS FEATURES (16 features total)
    # ========================================================================
    # Calculate Q_L from cavity voltage decay (if Ucav available)
    if 'Ucav' in signals:
        features['ql'], features['ql_tau'], features['ql_r2'] = calculate_ql(signals['Ucav'])
    else:
        features['ql'], features['ql_tau'], features['ql_r2'] = np.nan, np.nan, 0.0

    # Cavity detuning (phase derivative)
    if 'PhaseCav' in signals:
        phase_deriv = np.gradient(signals['PhaseCav'])
        features['detuning_hz'] = np.mean(phase_deriv) * 88e6 / (2 * np.pi)  # 88 MHz cavity
        features['detuning_bandwidth'] = features['detuning_hz'] / 88e6
    else:
        features['detuning_hz'], features['detuning_bandwidth'] = 0.0, 0.0

    # Control error (reference - cavity)
    if 'A Ucr' in signals and 'A Uamp' in signals:
        control_error = signals['A Ucr'] - signals['A Uamp']
        features['control_error_mean'] = np.mean(control_error)
        features['control_error_std'] = np.std(control_error)
        features['control_error_max'] = np.max(np.abs(control_error))

        # Settling time and overshoot (post-trigger analysis)
        trigger_idx = 3000
        if len(control_error) > trigger_idx + 500:
            error_post = control_error[trigger_idx:]

            # Overshoot: max error in first 500 samples post-trigger
            features['control_overshoot'] = np.max(np.abs(error_post[:500]))

            # Settling time: time to settle within 5% of final value
            try:
                final_val = np.median(error_post[-100:])
                threshold = 0.05 * np.abs(final_val) if final_val != 0 else 0.05
                settled = np.where(np.abs(error_post - final_val) < threshold)[0]
                features['control_settling_time'] = settled[0] / 88000.0 if len(settled) > 0 else np.inf
            except:
                features['control_settling_time'] = np.nan
        else:
            features['control_overshoot'] = 0.0
            features['control_settling_time'] = np.nan
    else:
        features['control_error_mean'], features['control_error_std'], features['control_error_max'] = 0.0, 0.0, 0.0
        features['control_overshoot'] = 0.0
        features['control_settling_time'] = np.nan

    # Phase stability (cavity)
    if 'PhaseCav' in signals:
        phase_cav = signals['PhaseCav']
        trigger_idx = 3000
        phase_pre = phase_cav[:trigger_idx] if len(phase_cav) > trigger_idx else phase_cav

        features['phase_stability_std'] = np.std(phase_pre)
        features['phase_stability_range'] = np.ptp(phase_pre)

        # Phase drift rate (linear trend)
        if len(phase_pre) > 1000:
            phase_unwrapped = np.unwrap(phase_pre)
            t = np.arange(len(phase_unwrapped)) / 88000.0
            try:
                coeffs = np.polyfit(t, phase_unwrapped, 1)
                features['phase_drift'] = coeffs[0]
            except:
                features['phase_drift'] = 0.0
        else:
            features['phase_drift'] = 0.0

        # FFT dominant frequency
        try:
            freqs = np.fft.rfftfreq(len(phase_pre), 1/88000.0)
            fft = np.abs(np.fft.rfft(phase_pre))
            features['phase_dom_freq_cav'] = freqs[1:][np.argmax(fft[1:])] if len(freqs) > 1 else 0.0
        except:
            features['phase_dom_freq_cav'] = 0.0
    else:
        features['phase_stability_std'], features['phase_stability_range'] = 0.0, 0.0
        features['phase_drift'] = 0.0
        features['phase_dom_freq_cav'] = 0.0

    # Phase stability (input)
    if 'PhaseUci' in signals:
        phase_uci = signals['PhaseUci']
        trigger_idx = 3000
        phase_pre_uci = phase_uci[:trigger_idx] if len(phase_uci) > trigger_idx else phase_uci

        # FFT dominant frequency
        try:
            freqs = np.fft.rfftfreq(len(phase_pre_uci), 1/88000.0)
            fft = np.abs(np.fft.rfft(phase_pre_uci))
            features['phase_dom_freq_uci'] = freqs[1:][np.argmax(fft[1:])] if len(freqs) > 1 else 0.0
        except:
            features['phase_dom_freq_uci'] = 0.0
    else:
        features['phase_dom_freq_uci'] = 0.0

    # Amplitude modulation depth
    if 'Ucav' in signals:
        ucav_abs = np.abs(signals['Ucav'])
        features['amplitude_mod_depth'] = (np.max(ucav_abs) - np.min(ucav_abs)) / np.mean(ucav_abs) * 100 if np.mean(ucav_abs) > 0 else 0
    else:
        features['amplitude_mod_depth'] = 0.0

    # Phase modulation depth
    if 'PhaseCav' in signals:
        features['phase_mod_depth'] = np.ptp(signals['PhaseCav'])
    else:
        features['phase_mod_depth'] = 0.0

    # Forward/reflected power ratio (if available)
    # CORRECTED: Use A Ucr (reflected signal amplitude) not courant pickup
    # Per thesis Table 2.2: A Ucr = "Amplitude du signal réfléchi"
    if 'Uci' in signals and 'A Ucr' in signals:
        forward = np.mean(np.abs(signals['Uci']))
        reflected = np.mean(np.abs(signals['A Ucr']))
        features['forward_reflected_ratio'] = forward / reflected if reflected > 0 else 0
    else:
        features['forward_reflected_ratio'] = 0.0

    # Cavity gradient
    if 'Ucav' in signals:
        features['cavity_gradient'] = np.mean(np.abs(signals['Ucav']))
    else:
        features['cavity_gradient'] = 0.0

    # ========================================================================
    # 3. PRECURSOR FEATURES (50+ features)
    # ========================================================================
    delta_pre = 3000

    for signal_name, signal_data in signals.items():
        if len(signal_data) < delta_pre:
            continue

        pre_trigger = signal_data[:delta_pre]

        # Temporal trends (linear fit)
        x = np.arange(len(pre_trigger))
        coeffs = np.polyfit(x, pre_trigger, 1)
        features[f'{signal_name}_trend_slope'] = coeffs[0]

        # Quadratic trend (acceleration)
        coeffs_quad = np.polyfit(x, pre_trigger, 2)
        features[f'{signal_name}_trend_acceleration'] = coeffs_quad[0]

        # CUSUM (cumulative sum for anomaly detection)
        cusum = np.cumsum(pre_trigger - np.mean(pre_trigger))
        features[f'{signal_name}_cusum_max'] = np.max(cusum)
        features[f'{signal_name}_cusum_min'] = np.min(cusum)

        # Multi-window comparison (early vs late)
        early_window = pre_trigger[:1000]
        late_window = pre_trigger[2000:3000]
        features[f'{signal_name}_early_late_diff'] = np.mean(late_window) - np.mean(early_window)
        features[f'{signal_name}_early_late_ratio'] = np.mean(late_window) / np.mean(early_window) if np.mean(early_window) != 0 else 0

        # Variance ratio (early vs late) - precursor indicator
        var_early = np.var(early_window)
        var_late = np.var(late_window)
        features[f'{signal_name}_var_ratio_late_early'] = var_late / (var_early + 1e-10)

    # ========================================================================
    # 4. DERIVATIVE FEATURES (velocity, acceleration)
    # ========================================================================
    for signal_name in signals.keys():
        if signal_name in first_deriv:
            # 1st derivative (velocity)
            vel = first_deriv[signal_name]
            features[f'{signal_name}_velocity_mean'] = np.mean(vel)
            features[f'{signal_name}_velocity_std'] = np.std(vel)
            features[f'{signal_name}_velocity_max'] = np.max(np.abs(vel))

        if signal_name in second_deriv:
            # 2nd derivative (acceleration)
            accel = second_deriv[signal_name]
            features[f'{signal_name}_accel_mean'] = np.mean(accel)
            features[f'{signal_name}_accel_std'] = np.std(accel)
            features[f'{signal_name}_accel_max'] = np.max(np.abs(accel))

    # ========================================================================
    # 5. SEGMENT-BASED FEATURES (5 segments per signal)
    # ========================================================================
    for signal_name, signal_segments in segments.items():
        for segment_name, segment_data in signal_segments.items():
            if len(segment_data) > 0:
                features[f'{signal_name}_segment_{segment_name}_mean'] = np.mean(segment_data)
                features[f'{signal_name}_segment_{segment_name}_std'] = np.std(segment_data)
                # Trend within segment
                x_seg = np.arange(len(segment_data))
                if len(x_seg) > 1:
                    coeffs_seg = np.polyfit(x_seg, segment_data, 1)
                    features[f'{signal_name}_segment_{segment_name}_slope'] = coeffs_seg[0]
                else:
                    features[f'{signal_name}_segment_{segment_name}_slope'] = 0.0

    # ========================================================================
    # 6. METADATA FEATURES (~10)
    # ========================================================================
    for key, value in metadata.items():
        if isinstance(value, (int, float)):
            features[f'meta_{key}'] = value
        elif isinstance(value, str):
            try:
                features[f'meta_{key}'] = float(value)
            except ValueError:
                pass

    return features


def calculate_ql(ucav_signal, fs=88000, decay_window_samples=500):
    """
    Calculate loaded quality factor Q_L from cavity voltage decay

    Args:
        ucav_signal: Cavity voltage time series (normalized)
        fs: Sampling frequency (Hz)
        decay_window_samples: Number of samples to use for decay fit

    Returns:
        (ql, tau, r2): Quality factor, decay time constant, fit R²
    """
    # Assume trigger at index 3000 (configurable)
    trigger_idx = 3000
    if len(ucav_signal) < trigger_idx + decay_window_samples:
        return np.nan, np.nan, 0.0

    decay_signal = ucav_signal[trigger_idx:trigger_idx + decay_window_samples]
    decay_signal = np.abs(decay_signal)

    # Ensure decay (not growth)
    if decay_signal[0] < decay_signal[-1]:
        return np.nan, np.nan, 0.0

    # Remove offset and avoid log(0)
    decay_signal = decay_signal - decay_signal.min() + 1e-10

    t = np.arange(len(decay_signal)) / fs

    try:
        # Fit exponential decay: A*exp(-t/tau)
        log_signal = np.log(decay_signal)
        coeffs = np.polyfit(t, log_signal, 1)
        tau = -1.0 / coeffs[0]

        # Calculate R² for fit quality
        y_pred = np.polyval(coeffs, t)
        ss_res = np.sum((log_signal - y_pred)**2)
        ss_tot = np.sum((log_signal - log_signal.mean())**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate Q_L from tau
        f0 = 88e6  # SPIRAL2 cavity frequency (88 MHz)
        ql = np.pi * f0 * tau

        return ql, tau, r2

    except Exception:
        return np.nan, np.nan, 0.0


class LLRFDataPreparation:
    """LLRF Data Preparation Pipeline for Cluster Processing"""

    def __init__(self, data_dir, output_dir, n_jobs=1, max_files=None, resume=False,
                 batch_size=2000, file_timeout=30):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs
        self.max_files = max_files
        self.resume = resume
        self.batch_size = batch_size
        self.file_timeout = file_timeout

        # Checkpoint files
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Number of jobs: {self.n_jobs}")
        logger.info(f"Max files: {self.max_files if self.max_files else 'All'}")
        logger.info(f"Resume: {self.resume}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"File timeout: {self.file_timeout}s")

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
                           '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.csv', '.fig']
        excluded_names = ['.directory', '.DS_Store', 'Thumbs.db']

        file_paths = []
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
            if not f.is_file():
                skip_stats['not_file'] += 1
                continue

            if f.name.startswith('.'):
                skip_stats['hidden'] += 1
                continue

            if f.name in excluded_names:
                skip_stats['excluded_name'] += 1
                continue

            if f.suffix.lower() in excluded_suffixes:
                skip_stats['excluded_suffix'] += 1
                continue

            try:
                file_stat = f.stat()
                if file_stat.st_size < 100:
                    skip_stats['too_small'] += 1
                    continue

                with open(f, 'rb') as test_f:
                    test_f.read(1)

            except PermissionError:
                skip_stats['permission_denied'] += 1
                continue
            except Exception:
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
        for key, val in skip_stats.items():
            logger.info(f"  - {key}: {val}")

        if file_paths:
            logger.info(f"Sample files found:")
            for f in file_paths[:5]:
                logger.info(f"  - {f.relative_to(self.data_dir)} ({f.stat().st_size} bytes)")

        # Save checkpoint
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(file_paths, f)
        logger.info(f"Saved file paths checkpoint: {checkpoint_file}")

        return file_paths

    def run_parallel(self, file_paths):
        """Run data preparation with multiprocessing and batching"""
        logger.info(f"Running parallel processing with {self.n_jobs} jobs...")
        logger.info(f"Processing {len(file_paths)} files in batches of {self.batch_size}")

        # Create batch directory
        batch_dir = self.output_dir / 'batches'
        batch_dir.mkdir(exist_ok=True)

        # Check for existing batches if resuming
        start_batch_idx = 0
        if self.resume:
            existing_batches = sorted(batch_dir.glob('batch_*.pkl'))
            if existing_batches:
                last_batch = existing_batches[-1]
                start_batch_idx = int(last_batch.stem.split('_')[1])
                logger.info(f"Resuming from batch {start_batch_idx} (found {len(existing_batches)} existing batches)")

        # Process in batches
        num_batches = (len(file_paths) + self.batch_size - 1) // self.batch_size
        all_batch_files = []

        # Track problematic files
        timeout_files = []
        error_files = []

        for batch_idx in range(start_batch_idx, num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(file_paths))
            batch_files = file_paths[start_idx:end_idx]

            logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_files)} files)...")

            # Create partial function with timeout parameter
            process_func = partial(process_single_file_with_timeout, file_timeout=self.file_timeout)

            with Pool(processes=self.n_jobs) as pool:
                batch_results = list(tqdm(
                    pool.imap(process_func, batch_files),
                    total=len(batch_files),
                    desc=f"Batch {batch_idx + 1}/{num_batches}"
                ))

            # Track failures
            for result in batch_results:
                if result and not result['success']:
                    if 'TIMEOUT' in result.get('error', ''):
                        timeout_files.append(result['file'])
                    else:
                        error_files.append((result['file'], result.get('error', 'Unknown')))

            # Filter successful results
            batch_results = [r for r in batch_results if r and r['success']]

            # Save batch immediately
            batch_file = batch_dir / f'batch_{batch_idx:03d}.pkl'
            with open(batch_file, 'wb') as f:
                pickle.dump({
                    'results': batch_results,
                    'n_events': len(batch_results),
                    'batch_idx': batch_idx
                }, f)

            logger.info(f"Saved batch {batch_idx + 1}: {len(batch_results)} events → {batch_file}")
            all_batch_files.append(batch_file)

            # Free memory
            del batch_results

        # Save problematic files list
        if timeout_files:
            timeout_file = self.output_dir / 'timeout_files.txt'
            with open(timeout_file, 'w') as f:
                f.write(f"Files that timed out (>{self.file_timeout}s)\n")
                f.write("=" * 80 + "\n\n")
                for file in timeout_files:
                    f.write(f"{file}\n")
            logger.warning(f"⚠️  {len(timeout_files)} files timed out. List saved to: {timeout_file}")

        # Load and merge all batches
        logger.info("Merging all batches...")
        all_results = []
        for batch_file in sorted(batch_dir.glob('batch_*.pkl')):
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
                all_results.extend(batch_data['results'])
                logger.info(f"Loaded {batch_data['n_events']} events from {batch_file.name}")

        logger.info(f"Total events from all batches: {len(all_results)}")
        return all_results

    def save_results(self, results):
        """Save processed results to disk"""
        logger.info(f"Saving {len(results)} processed events...")

        if len(results) == 0:
            logger.error("No results to save - all files failed to process")
            raise ValueError("No data was successfully processed")

        # Separate components
        metadata_list = [r['metadata'] for r in results]
        features_list = [r['features'] for r in results]
        fault_labels = [r['fault_labels'] for r in results]

        # Create feature dataframe
        df_features = pd.DataFrame(features_list)

        # Create labels
        y_binary = np.array([f['binary'] for f in fault_labels])
        y_multilabel = np.array([f['multilabel'] for f in fault_labels])

        # Log fault vs no-fault distribution
        n_normal = np.sum(y_binary == 0)
        n_fault = np.sum(y_binary == 1)
        logger.info(f"")
        logger.info(f"="*80)
        logger.info(f"DATASET COMPOSITION (Critical for ML Training):")
        logger.info(f"="*80)
        logger.info(f"  Total events:    {len(y_binary)}")
        logger.info(f"  Normal (ALM=0):  {n_normal} ({100*n_normal/len(y_binary):.1f}%)")
        logger.info(f"  Fault (ALM>0):   {n_fault} ({100*n_fault/len(y_binary):.1f}%)")
        logger.info(f"  Balance ratio:   {n_fault}/{n_normal} = {n_fault/(n_normal+1e-10):.2f}:1")
        logger.info(f"")
        if n_normal == 0:
            logger.warning(f"⚠️  WARNING: NO NORMAL EVENTS DETECTED!")
            logger.warning(f"  ML training requires BOTH fault and no-fault events!")
            logger.warning(f"  Check if manual acquisitions are in the dataset")
        if n_fault == 0:
            logger.warning(f"⚠️  WARNING: NO FAULT EVENTS DETECTED!")
        logger.info(f"="*80)

        # Save preprocessed data
        preprocessed_file = self.output_dir / 'preprocessed_data.pkl'

        # Extract actual signal names (not the processed structure keys)
        signal_names = list(results[0]['signals']['signals'].keys()) if results else []

        with open(preprocessed_file, 'wb') as f:
            pickle.dump({
                'signals_normalized': [r['signals'] for r in results],
                'metadata': metadata_list,
                'signal_names': signal_names,
            }, f)
        logger.info(f"Saved preprocessed data: {preprocessed_file}")

        # Save features
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # Step 1: Correlation filtering - remove highly correlated features
        logger.info("Applying feature selection (correlation filtering)...")
        feature_cols = df_features.columns.tolist()
        X_features = df_features.fillna(0).values

        corr_matrix = np.corrcoef(X_features.T)
        threshold = 0.95
        to_drop = set()

        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > threshold:
                    to_drop.add(feature_cols[j])

        logger.info(f"Features to drop (corr > {threshold}): {len(to_drop)}/{len(feature_cols)}")

        feature_cols_filtered = [col for col in feature_cols if col not in to_drop]
        df_features_filtered = df_features[feature_cols_filtered]
        X_filtered = df_features_filtered.fillna(0).values

        logger.info(f"Remaining features after correlation filtering: {len(feature_cols_filtered)}")

        # Step 2: Standardization (zero mean, unit variance)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)

        # Step 3: PCA dimensionality reduction (95% variance retention)
        n_samples, n_features = X_scaled.shape
        if n_samples > 1 and n_features > 1:
            pca = PCA(n_components=0.95, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            logger.info(f"PCA results:")
            logger.info(f"  Original features: {X_scaled.shape[1]}")
            logger.info(f"  PCA components: {X_pca.shape[1]}")
            logger.info(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")
        else:
            logger.warning(f"Not enough samples/features for PCA ({n_samples} samples, {n_features} features)")
            pca = None
            X_pca = X_scaled

        # ========================================================================
        # Reconstruct sequence arrays for temporal models (Step 05 precursor detection)
        # ========================================================================
        logger.info("Reconstructing sequence arrays for temporal/sequence-based models...")

        n_events = len(results)
        n_samples = 4000  # Standard LLRF acquisition length
        n_signals = len(signal_names)

        # Initialize arrays
        sequences_full = np.zeros((n_events, n_samples, n_signals))
        sequences_pretrigger = np.zeros((n_events, 3000, n_signals))  # Pre-trigger only
        sequences_downsampled = np.zeros((n_events, n_samples // 10, n_signals))  # Downsample by 10

        # Fill arrays from results
        for i, result in enumerate(results):
            signals_dict = result['signals']['signals']  # Access normalized signals

            for j, signal_name in enumerate(signal_names):
                if signal_name in signals_dict:
                    sig = signals_dict[signal_name]
                    sig_len = min(len(sig), n_samples)

                    # Full sequence (zero-padded if shorter than 4000)
                    sequences_full[i, :sig_len, j] = sig[:sig_len]

                    # Pre-trigger portion (first 3000 samples)
                    pretrig_len = min(len(sig), 3000)
                    sequences_pretrigger[i, :pretrig_len, j] = sig[:pretrig_len]

                    # Downsampled (every 10th sample)
                    downsampled = sig[::10]
                    downsamp_len = min(len(downsampled), n_samples // 10)
                    sequences_downsampled[i, :downsamp_len, j] = downsampled[:downsamp_len]

        logger.info(f"Sequence arrays reconstructed:")
        logger.info(f"  sequences_full: {sequences_full.shape}")
        logger.info(f"  sequences_pretrigger: {sequences_pretrigger.shape}")
        logger.info(f"  sequences_downsampled: {sequences_downsampled.shape}")

        features_file = self.output_dir / 'features_engineered.pkl'
        with open(features_file, 'wb') as f:
            pickle.dump({
                # Feature matrices
                'features_all': df_features,
                'feature_cols': feature_cols_filtered,  # Filtered features (after correlation removal)
                'X_scaled': X_scaled,
                'X_pca': X_pca,

                # Labels
                'y_binary': y_binary,
                'y_multilabel': y_multilabel,
                'fault_column_names': [f'Fault_{i}' for i in range(7)],

                # Transformers
                'scaler': scaler,
                'pca': pca,

                # Metadata
                'metadata': metadata_list,
                'signal_names': signal_names,

                # Sequence arrays for temporal/sequence models (Step 05 precursor detection)
                'sequences_full': sequences_full,
                'sequences_pretrigger': sequences_pretrigger,
                'sequences_downsampled': sequences_downsampled,
            }, f)
        logger.info(f"Saved features: {features_file}")
        logger.info(f"  Including {n_signals} signal channels × {n_events} events")
        logger.info(f"  Sequence arrays: full={sequences_full.shape}, pretrig={sequences_pretrigger.shape}, downsampled={sequences_downsampled.shape}")

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

        process = psutil.Process()
        logger.info(f"Initial memory usage: {process.memory_info().rss / 1e9:.2f} GB")

        # Step 1: Collect file paths
        file_paths = self.collect_file_paths()

        # Step 2: Process files
        if self.n_jobs > 1:
            results = self.run_parallel(file_paths)
        else:
            logger.error("Single-process mode not supported in this version. Use --n-jobs > 1")
            raise ValueError("Use --n-jobs > 1")

        logger.info(f"Successfully processed {len(results)}/{len(file_paths)} files")

        if len(results) == 0:
            logger.error("FATAL: No files were successfully processed!")
            raise RuntimeError("No data was successfully processed")

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
        description='LLRF Data Preparation for Cluster (WITH TIMEOUT PROTECTION)',
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
                        help='Resume from last successful batch')
    parser.add_argument('--batch-size', type=int, default=2000,
                        help='Batch size for processing (default: 2000)')
    parser.add_argument('--file-timeout', type=int, default=30,
                        help='Timeout per file in seconds (default: 30s)')

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
        batch_size=args.batch_size,
        file_timeout=args.file_timeout
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
