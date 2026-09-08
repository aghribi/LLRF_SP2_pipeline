#!/usr/bin/env python3
"""
Step 03: Feature Engineering with Precursor Analysis

Description: Extracts comprehensive features from preprocessed LLRF signals including
             domain-specific physics features, statistical features, and precursor
             signatures for early fault detection.

Input:  cooked_data/step_02_preprocessing/preprocessed_data.pkl
        cooked_data/step_01_loading/fault_labels_matrix.npy (multi-label targets)

Output: cooked_data/step_03_features/features_engineered.pkl

Features Extracted:
1. Domain-specific physics (16): Q_L, detuning, control error, phase stability
2. Statistical (112): mean, std, peaks, moments, etc. for 8 signals
3. Precursor (50+): Temporal trends, CUSUM, derivatives, multi-window comparisons

Usage:
    python prepare_03_features.py \
        --input /path/to/step_02_preprocessing \
        --output /path/to/step_03_features \
        --step01-dir /path/to/step_01_loading
"""

import sys
import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step_03_features.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Step03Processor:
    """Processor for Step 03: Feature Engineering"""

    def __init__(self, input_dir, output_dir, step01_dir=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.step01_dir = Path(step01_dir) if step01_dir else None
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.data = None
        self.signals_normalized = None
        self.first_deriv_normalized = None
        self.second_deriv_normalized = None
        self.segmented_signals = None
        self.metadata_df = None
        self.signal_names = None
        self.config = None
        self.fault_labels_matrix = None
        self.fault_column_names = None

    # ========================================
    # Domain Feature Calculation Functions
    # ========================================

    def calculate_ql(self, ucav_signal, fs=88000, decay_window_samples=500):
        """Calculate quality factor Q_L from cavity voltage decay."""
        trigger_idx = self.config['delta_pre']
        decay_signal = ucav_signal[trigger_idx:trigger_idx + decay_window_samples]
        decay_signal = np.abs(decay_signal)
        decay_signal = decay_signal - decay_signal.min() + 1e-10

        if decay_signal[0] < decay_signal[-1]:
            return np.nan, np.nan, 0.0

        t = np.arange(len(decay_signal)) / fs

        try:
            log_signal = np.log(decay_signal)
            coeffs = np.polyfit(t, log_signal, 1)
            tau = -1.0 / coeffs[0]

            y_pred = np.polyval(coeffs, t)
            ss_res = np.sum((log_signal - y_pred)**2)
            ss_tot = np.sum((log_signal - log_signal.mean())**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            f0 = 88e6  # SPIRAL2 cavity frequency
            ql = np.pi * f0 * tau

            return ql, tau, r2
        except:
            return np.nan, np.nan, 0.0

    def calculate_detuning(self, phase_signal, fs=88000, window_samples=1000):
        """Estimate detuning from phase drift rate."""
        trigger_idx = self.config['delta_pre']
        phase_window = phase_signal[trigger_idx-window_samples:trigger_idx]
        phase_unwrapped = np.unwrap(phase_window)
        t = np.arange(len(phase_unwrapped)) / fs

        try:
            coeffs = np.polyfit(t, phase_unwrapped, 1)
            phase_drift_rate = coeffs[0]
            detuning = phase_drift_rate / (2 * np.pi)
            return detuning, phase_drift_rate
        except:
            return 0.0, 0.0

    def calculate_control_error(self, ucav, ucr, uci, fs=88000):
        """Calculate control loop error metrics."""
        trigger_idx = self.config['delta_pre']
        error = ucav - ucr
        error_pre = error[:trigger_idx]
        error_post = error[trigger_idx:]

        rms_pre = np.sqrt(np.mean(error_pre**2))
        rms_post = np.sqrt(np.mean(error_post**2))
        max_error = np.max(np.abs(error))

        try:
            final_val = np.median(error_post[-100:])
            threshold = 0.05 * np.abs(final_val)
            settled = np.where(np.abs(error_post - final_val) < threshold)[0]
            settling_time = settled[0] / fs if len(settled) > 0 else np.inf
        except:
            settling_time = np.nan

        overshoot = np.max(np.abs(error_post[:500])) if len(error_post) > 500 else np.max(np.abs(error_post))

        return rms_pre, rms_post, max_error, settling_time, overshoot

    def calculate_phase_stability(self, phase_signal, fs=88000):
        """Calculate phase stability metrics."""
        trigger_idx = self.config['delta_pre']
        phase_pre = phase_signal[:trigger_idx]

        jitter = np.std(phase_pre)
        excursion = np.ptp(phase_pre)

        try:
            freqs = np.fft.rfftfreq(len(phase_pre), 1/fs)
            fft = np.abs(np.fft.rfft(phase_pre))
            dominant_freq = freqs[1:][np.argmax(fft[1:])]
        except:
            dominant_freq = 0.0

        return jitter, excursion, dominant_freq

    # ========================================
    # Precursor Feature Functions
    # ========================================

    def calculate_temporal_trends(self, seg_data_all, signal_idx):
        """Calculate linear and quadratic trends across temporal segments."""
        slopes = []
        accels = []

        for event_idx in range(seg_data_all.shape[0]):
            seg_signal = seg_data_all[event_idx, :, signal_idx]
            t = np.arange(len(seg_signal))

            try:
                coeffs_lin = np.polyfit(t, seg_signal, 1)
                slope = coeffs_lin[0]
            except:
                slope = 0.0

            try:
                coeffs_quad = np.polyfit(t, seg_signal, 2)
                accel = coeffs_quad[0]
            except:
                accel = 0.0

            slopes.append(slope)
            accels.append(accel)

        return np.array(slopes), np.array(accels)

    def calculate_cusum_features(self, signals_pretrigger, baseline_samples=600):
        """Calculate CUSUM statistics for change point detection."""
        features = {}
        n_events, n_samples, n_signals = signals_pretrigger.shape

        for sig_idx, sig_name in enumerate(self.signal_names):
            cusum_max_vals = []

            for event_idx in range(n_events):
                signal_arr = signals_pretrigger[event_idx, :, sig_idx]
                baseline = signal_arr[:baseline_samples]
                mu = np.mean(baseline)
                sigma = np.std(baseline)
                K = 0.5 * sigma

                S = np.zeros(len(signal_arr))
                for k in range(1, len(signal_arr)):
                    S[k] = max(0, S[k-1] + (signal_arr[k] - mu - K))

                cusum_max_vals.append(np.max(S))

            features[f'{sig_name}_cusum_max'] = np.array(cusum_max_vals)

        return features

    def calculate_derivative_features(self):
        """Extract rate-of-change features from derivatives."""
        features = {}

        if self.first_deriv_normalized is None or self.second_deriv_normalized is None:
            logger.warning("Derivatives not available")
            return features

        for sig_idx, sig_name in enumerate(self.signal_names):
            first_deriv_abs = np.abs(self.first_deriv_normalized[:, :, sig_idx])
            features[f'{sig_name}_deriv1_max'] = np.max(first_deriv_abs, axis=1)
            features[f'{sig_name}_deriv1_mean'] = np.mean(first_deriv_abs, axis=1)

            second_deriv_abs = np.abs(self.second_deriv_normalized[:, :, sig_idx])
            features[f'{sig_name}_deriv2_max'] = np.max(second_deriv_abs, axis=1)
            features[f'{sig_name}_deriv2_mean'] = np.mean(second_deriv_abs, axis=1)

        return features

    def calculate_multiwindow_features(self):
        """Compare early vs late segment statistics."""
        features = {}

        if 'early' not in self.segmented_signals or 'late' not in self.segmented_signals:
            return features

        early_seg = self.segmented_signals['early']
        late_seg = self.segmented_signals['late']

        for sig_idx, sig_name in enumerate(self.signal_names):
            var_ratios = []

            for event_idx in range(early_seg.shape[0]):
                early_signal = early_seg[event_idx, :, sig_idx]
                late_signal = late_seg[event_idx, :, sig_idx]

                var_early = np.var(early_signal)
                var_late = np.var(late_signal)
                var_ratio = var_late / (var_early + 1e-10)
                var_ratios.append(var_ratio)

            features[f'{sig_name}_var_ratio_late_early'] = np.array(var_ratios)

        return features

    @staticmethod
    def extract_statistical_features(signal, prefix=''):
        """Extract statistical features from 1D signal."""
        features = {}

        features[f'{prefix}mean'] = np.mean(signal)
        features[f'{prefix}std'] = np.std(signal)
        features[f'{prefix}min'] = np.min(signal)
        features[f'{prefix}max'] = np.max(signal)
        features[f'{prefix}median'] = np.median(signal)
        features[f'{prefix}iqr'] = np.percentile(signal, 75) - np.percentile(signal, 25)
        features[f'{prefix}range'] = np.ptp(signal)
        features[f'{prefix}skewness'] = stats.skew(signal)
        features[f'{prefix}kurtosis'] = stats.kurtosis(signal)
        features[f'{prefix}energy'] = np.sum(signal**2)
        features[f'{prefix}rms'] = np.sqrt(np.mean(signal**2))

        peaks, _ = find_peaks(signal, distance=20)
        features[f'{prefix}n_peaks'] = len(peaks)

        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        features[f'{prefix}n_zero_crossings'] = len(zero_crossings)

        if len(signal) > 1:
            features[f'{prefix}autocorr_lag1'] = np.corrcoef(signal[:-1], signal[1:])[0, 1]
        else:
            features[f'{prefix}autocorr_lag1'] = 0.0

        return features

    def load_inputs(self):
        """Load preprocessed data from Step 02 and fault labels from Step 01"""
        logger.info("Loading inputs...")

        # Load preprocessed data
        preprocessed_file = self.input_dir / 'preprocessed_data.pkl'
        if not preprocessed_file.exists():
            raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_file}")

        with open(preprocessed_file, 'rb') as f:
            self.data = pickle.load(f)

        self.signals_normalized = self.data['signals_normalized']
        self.first_deriv_normalized = self.data.get('first_derivative_normalized', None)
        self.second_deriv_normalized = self.data.get('second_derivative_normalized', None)
        self.segmented_signals = self.data.get('segmented_signals', {})
        self.metadata_df = self.data['metadata']
        self.signal_names = self.data['signal_names']
        self.config = self.data['config']

        n_events = self.signals_normalized.shape[0]

        logger.info(f"Loaded preprocessed data:")
        logger.info(f"  Events: {n_events}")
        logger.info(f"  Signal channels: {len(self.signal_names)}")
        logger.info(f"  Derivatives available: {self.first_deriv_normalized is not None}")
        logger.info(f"  Temporal segments: {len(self.segmented_signals)}")

        # Load fault labels
        if self.step01_dir:
            fault_matrix_file = self.step01_dir / 'fault_labels_matrix.npy'
            fault_names_file = self.step01_dir / 'fault_column_names.pkl'

            if fault_matrix_file.exists() and fault_names_file.exists():
                self.fault_labels_matrix = np.load(fault_matrix_file)
                with open(fault_names_file, 'rb') as f:
                    self.fault_column_names = pickle.load(f)

                logger.info(f"Loaded fault labels:")
                logger.info(f"  Shape: {self.fault_labels_matrix.shape}")
                logger.info(f"  Fault types: {self.fault_column_names}")
            else:
                logger.warning("Fault labels not found")
        else:
            logger.warning("Step01 directory not provided - fault labels unavailable")

    def process(self):
        """Main feature extraction logic"""
        logger.info("="*70)
        logger.info("FEATURE ENGINEERING")
        logger.info("="*70)

        n_events = self.signals_normalized.shape[0]

        logger.info("Extracting features for all events...")

        all_features_list = []

        for i in tqdm(range(n_events), desc="Events"):
            event_features = {}

            # Get signal indices
            ucav_idx = self.signal_names.index('Ucav')
            phase_cav_idx = self.signal_names.index('PhaseCav')
            uci_idx = self.signal_names.index('Uci')
            phase_uci_idx = self.signal_names.index('PhaseUci')
            ucr_idx = self.signal_names.index('A Ucr')

            # Extract signals
            ucav = self.signals_normalized[i, :, ucav_idx]
            phase_cav = self.signals_normalized[i, :, phase_cav_idx]
            uci = self.signals_normalized[i, :, uci_idx]
            phase_uci = self.signals_normalized[i, :, phase_uci_idx]
            ucr = self.signals_normalized[i, :, ucr_idx]

            # Domain features
            ql, tau, r2 = self.calculate_ql(ucav)
            event_features['ql'] = ql
            event_features['decay_tau'] = tau
            event_features['ql_fit_r2'] = r2

            detuning, drift = self.calculate_detuning(phase_cav)
            event_features['detuning_hz'] = detuning
            event_features['phase_drift'] = drift

            rms_pre, rms_post, max_err, settle, overshoot = self.calculate_control_error(ucav, ucr, uci)
            event_features['control_rms_pre'] = rms_pre
            event_features['control_rms_post'] = rms_post
            event_features['control_max_error'] = max_err
            event_features['control_settling_time'] = settle
            event_features['control_overshoot'] = overshoot

            jitter_cav, excursion_cav, dom_freq_cav = self.calculate_phase_stability(phase_cav)
            event_features['phase_jitter_cav'] = jitter_cav
            event_features['phase_excursion_cav'] = excursion_cav
            event_features['phase_dom_freq_cav'] = dom_freq_cav

            jitter_uci, excursion_uci, dom_freq_uci = self.calculate_phase_stability(phase_uci)
            event_features['phase_jitter_uci'] = jitter_uci
            event_features['phase_excursion_uci'] = excursion_uci
            event_features['phase_dom_freq_uci'] = dom_freq_uci

            # Statistical features for all signals
            for j, sig_name in enumerate(self.signal_names):
                sig = self.signals_normalized[i, :, j]
                sig_features = self.extract_statistical_features(sig, prefix=f'{sig_name}_')
                event_features.update(sig_features)

            all_features_list.append(event_features)

        # Convert to DataFrame
        df_features_base = pd.DataFrame(all_features_list)
        logger.info(f"Extracted base features: {len(df_features_base.columns)}")

        # Add precursor features
        precursor_features_dict = {}

        # Temporal trends for key signals
        for sig_name in ['Ucav', 'PhaseCav']:
            if sig_name in self.signal_names:
                sig_idx = self.signal_names.index(sig_name)
                segment_names = ['early', 'mid_early', 'mid', 'mid_late', 'late']

                for seg_name in segment_names:
                    if seg_name in self.segmented_signals:
                        slopes, accels = self.calculate_temporal_trends(
                            self.segmented_signals[seg_name], sig_idx
                        )
                        precursor_features_dict[f'{sig_name}_{seg_name}_slope'] = slopes
                        precursor_features_dict[f'{sig_name}_{seg_name}_accel'] = accels

        # CUSUM
        pretrigger_signals = self.segmented_signals.get(
            'full_pretrigger',
            self.signals_normalized[:, :self.config['delta_pre'], :]
        )
        cusum_feats = self.calculate_cusum_features(pretrigger_signals)
        precursor_features_dict.update(cusum_feats)

        # Derivatives
        deriv_feats = self.calculate_derivative_features()
        precursor_features_dict.update(deriv_feats)

        # Multi-window
        multiwin_feats = self.calculate_multiwindow_features()
        precursor_features_dict.update(multiwin_feats)

        df_precursor = pd.DataFrame(precursor_features_dict)
        logger.info(f"Extracted precursor features: {len(df_precursor.columns)}")

        # Combine all features
        df_features_all = pd.concat([df_features_base, df_precursor], axis=1)

        # Add metadata
        event_ids = [meta['event_id'] for meta in self.metadata_df[:n_events]]
        cavities = [meta['original_row']['ID'] for meta in self.metadata_df[:n_events]]

        df_features_all['event_id'] = event_ids
        df_features_all['cavity'] = cavities

        logger.info(f"Total features: {df_features_all.shape[1] - 2} (+ 2 metadata)")

        # Handle missing values
        n_missing = df_features_all.isnull().sum().sum()
        if n_missing > 0:
            logger.info(f"Filling {n_missing} missing values with median...")
            numeric_cols = df_features_all.select_dtypes(include=[np.number]).columns
            df_features_all[numeric_cols] = df_features_all[numeric_cols].fillna(
                df_features_all[numeric_cols].median()
            )

        # Feature selection - correlation filtering
        feature_cols = [col for col in df_features_all.columns if col not in ['event_id', 'cavity']]
        X_features = df_features_all[feature_cols].values

        corr_matrix = np.corrcoef(X_features.T)
        threshold = 0.95
        to_drop = set()

        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > threshold:
                    to_drop.add(feature_cols[j])

        logger.info(f"Features to drop (corr > {threshold}): {len(to_drop)}")

        feature_cols_filtered = [col for col in feature_cols if col not in to_drop]
        X_filtered = df_features_all[feature_cols_filtered].values

        logger.info(f"Remaining features: {len(feature_cols_filtered)}")

        # Standardization and PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)

        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        logger.info(f"PCA results:")
        logger.info(f"  Original features: {X_scaled.shape[1]}")
        logger.info(f"  PCA components: {X_pca.shape[1]}")
        logger.info(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

        # Prepare output data
        feature_data = {
            'features_all': df_features_all,
            'feature_cols': feature_cols_filtered,
            'X_scaled': X_scaled,
            'X_pca': X_pca,

            # Sequences for deep learning
            'sequences_full': self.signals_normalized,
            'sequences_pretrigger': pretrigger_signals,
            'sequences_downsampled': self.signals_normalized[:, ::10, :],

            # Multi-label targets
            'y_binary': (self.fault_labels_matrix.sum(axis=1) > 0).astype(int) if self.fault_labels_matrix is not None else None,
            'y_multilabel': self.fault_labels_matrix,
            'fault_column_names': self.fault_column_names,

            # Transformers and metadata
            'scaler': scaler,
            'pca': pca,
            'metadata': self.metadata_df[:n_events],
            'signal_names': self.signal_names,
        }

        return feature_data

    def save_outputs(self, results):
        """Save feature data"""
        logger.info("")
        logger.info("Saving feature data...")

        output_file = self.output_dir / 'features_engineered.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f, protocol=4)

        file_size_mb = output_file.stat().st_size / (1024**2)

        logger.info("="*70)
        logger.info("SAVED FEATURE DATA")
        logger.info("="*70)
        logger.info(f"File: {output_file}")
        logger.info(f"Size: {file_size_mb:.2f} MB")
        logger.info("")
        logger.info("Contents:")
        logger.info(f"  - features_all: {results['features_all'].shape}")
        logger.info(f"  - X_scaled: {results['X_scaled'].shape}")
        logger.info(f"  - X_pca: {results['X_pca'].shape}")
        logger.info(f"  - sequences_full: {results['sequences_full'].shape}")
        logger.info(f"  - y_binary: {results['y_binary'].shape if results['y_binary'] is not None else 'N/A'}")
        logger.info(f"  - y_multilabel: {results['y_multilabel'].shape if results['y_multilabel'] is not None else 'N/A'}")
        logger.info("")
        logger.info("✓ Ready for multi-phase anomaly detection pipeline!")

    def run(self):
        """Execute full feature engineering pipeline"""
        logger.info("="*70)
        logger.info("STEP 03: FEATURE ENGINEERING")
        logger.info("="*70)
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("")

        self.load_inputs()
        results = self.process()
        self.save_outputs(results)

        logger.info("")
        logger.info("="*70)
        logger.info("STEP 03 COMPLETE!")
        logger.info("="*70)

        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Step 03: Feature Engineering')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory (step_02_preprocessing)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory (step_03_features)')
    parser.add_argument('--step01-dir', type=str, default=None,
                        help='Step 01 directory for fault labels (optional)')
    args = parser.parse_args()

    processor = Step03Processor(args.input, args.output, args.step01_dir)
    processor.run()


if __name__ == '__main__':
    main()
