# Detailed Pipeline Comparison Analysis
## Original Modular Pipeline vs Enhanced Monolithic Pipeline

**Date**: 2025-12-27 10:20
**Purpose**: Pre-deployment analysis to ensure feature parity

---

## Executive Summary

### Critical Finding: Signal Channel Difference

| Aspect | Original Pipeline | Enhanced Pipeline |
|--------|------------------|-------------------|
| **Signal Channels** | **8 primary channels** (filtered) | **27 channels** (all available) |
| **Total Features** | ~180 features | **1,157 features** |
| **Feature Types** | Complete set | **Missing some key features** |

**Verdict**: Enhanced pipeline extracts MORE signals but is MISSING some important feature calculations.

---

## 1. Signal Extraction Comparison

### Original Pipeline (Step 02) - 8 Signals Only

```python
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
```

**Rationale**: These 8 signals are the PRIMARY physics channels for LLRF analysis.

### Enhanced Pipeline - 27 Signals (All Available)

Extracts ALL signals from PyPostMortem:
- I/Q pairs: `I Ucav binary`, `Q Ucav binary`, `I Fref binary`, `Q Fref binary`, etc.
- Amplitude signals: `A Ucr binary`, `A Uamp binary`
- Phase signals: `PhaseCav`, `PhaseUci`
- Additional: `vide`, `courant pickup`, `I modulateur binary`, `Q modulateur binary`, etc.

**Result**: 27 signals × features per signal = **1,157 total features**

---

## 2. Feature Engineering Comparison

### 2.1 Statistical Features

#### Original Pipeline (14 per signal)
```python
✅ mean
✅ std
✅ min
✅ max
✅ median
✅ iqr (q75 - q25)
✅ range (ptp)
✅ skewness
✅ kurtosis
✅ energy         # sum(signal²) - MISSING IN ENHANCED
✅ rms
✅ n_peaks
✅ n_zero_crossings  # MISSING IN ENHANCED
✅ autocorr_lag1     # MISSING IN ENHANCED
```

**Enhanced Pipeline has**: mean, std, min, max, range, median, q1, q3, iqr, skewness, kurtosis, rms, crest_factor, peak_count

**MISSING**: `energy`, `n_zero_crossings`, `autocorr_lag1`

---

### 2.2 Physics Features

#### Original Pipeline (16 features)
```python
# Q_L calculation
✅ ql
✅ decay_tau
✅ ql_fit_r2

# Detuning (from phase drift)
✅ detuning_hz
✅ phase_drift

# Control error (sophisticated)
✅ control_rms_pre
✅ control_rms_post
✅ control_max_error
✅ control_settling_time  # MISSING IN ENHANCED
✅ control_overshoot      # MISSING IN ENHANCED

# Phase stability (with FFT)
✅ phase_jitter_cav
✅ phase_excursion_cav
✅ phase_dom_freq_cav     # FFT dominant frequency - MISSING IN ENHANCED
✅ phase_jitter_uci
✅ phase_excursion_uci
✅ phase_dom_freq_uci     # FFT dominant frequency - MISSING IN ENHANCED
```

**Enhanced Pipeline has**: ql, ql_tau, ql_r2, detuning_hz, detuning_bandwidth, control_error_mean, control_error_std, control_error_max, phase_stability_std, phase_stability_range, amplitude_mod_depth, phase_mod_depth, forward_reflected_ratio, cavity_gradient

**MISSING**:
- `control_settling_time` (time to settle to final value)
- `control_overshoot` (max error in first 500 samples post-trigger)
- `phase_dom_freq_cav/uci` (FFT dominant frequency for phase jitter)
- `phase_drift` (rate of phase change)

**EXTRA** (not in original):
- `detuning_bandwidth` (normalized detuning)
- `amplitude_mod_depth`
- `phase_mod_depth`
- `forward_reflected_ratio`
- `cavity_gradient`

---

### 2.3 Precursor Features

#### Original Pipeline
```python
# Temporal trends (per segment, per signal)
✅ {signal}_{segment}_slope     # Linear trend
✅ {signal}_{segment}_accel     # Quadratic trend

# CUSUM
✅ {signal}_cusum_max

# Multi-window comparison
✅ {signal}_var_ratio_late_early  # Variance ratio - MISSING IN ENHANCED

# Derivatives
✅ {signal}_deriv1_max
✅ {signal}_deriv1_mean
✅ {signal}_deriv2_max
✅ {signal}_deriv2_mean
```

**Enhanced Pipeline has**:
- `{signal}_trend_slope`, `{signal}_trend_acceleration`
- `{signal}_cusum_max`, `{signal}_cusum_min`
- `{signal}_early_late_diff`, `{signal}_early_late_ratio`
- `{signal}_velocity_mean/std/max`
- `{signal}_accel_mean/std/max`
- `{signal}_segment_{name}_mean/std/slope` (for 5 segments)

**MISSING**:
- Per-segment trends (slope/accel) for EACH of 5 segments for EACH signal
- `var_ratio_late_early` (variance ratio between early and late windows)

**DIFFERENCES**:
- Original: Calculates segment trends only for KEY signals (`Ucav`, `PhaseCav`)
- Enhanced: Calculates trend_slope/acceleration on full pre-trigger, NOT per segment
- Enhanced: Calculates segment features (mean/std/slope) for ALL segments

---

## 3. Feature Selection & Dimensionality Reduction

### Both Pipelines - IDENTICAL

| Step | Original | Enhanced |
|------|----------|----------|
| **Correlation filtering** | threshold = 0.95 | threshold = 0.95 |
| **StandardScaler** | ✅ Yes | ✅ Yes |
| **PCA** | 95% variance | 95% variance |
| **Output structure** | Same format | Same format |

---

## 4. Data Structure Comparison

### Original Pipeline (Step 02 output)

```python
{
    'signals_normalized': ndarray,  # (n_events, 4000, 8)
    'first_derivative_normalized': ndarray,  # (n_events, 4000, 8)
    'second_derivative_normalized': ndarray,  # (n_events, 4000, 8)
    'segmented_signals': {
        'early': ndarray,  # (n_events, 600, 8)
        'mid_early': ndarray,
        'mid': ndarray,
        'mid_late': ndarray,
        'late': ndarray,
        'full_pretrigger': ndarray  # (n_events, 3000, 8)
    },
    'metadata': list of dicts,
    'signal_names': ['A Ucr', 'A Uamp', 'vide', ..., 'PhaseUci'],
    'config': {...}
}
```

### Enhanced Pipeline (preprocessed_data.pkl)

```python
{
    'signals_normalized': list of dicts,  # Each element is processed signals dict
    'metadata': list of dicts,
    'signal_names': ['I Ucav binary', 'Q Ucav binary', ..., ...]  # 27 signals
}

# Each signals dict in signals_normalized:
{
    'signals': {signal_name: ndarray, ...},  # 27 signals
    'first_deriv': {signal_name: ndarray, ...},
    'second_deriv': {signal_name: ndarray, ...},
    'segments': {
        signal_name: {
            'early': ndarray,
            'mid_early': ndarray,
            'mid': ndarray,
            'mid_late': ndarray,
            'late': ndarray
        }, ...
    }
}
```

**MAJOR STRUCTURAL DIFFERENCE**:
- Original: Single large numpy arrays (n_events, n_samples, n_signals)
- Enhanced: List of dicts (more flexible but less efficient for batch operations)

---

## 5. Missing Features Summary

### Critical Missing Features in Enhanced Pipeline

1. **Statistical**:
   - `energy` (sum of signal²)
   - `n_zero_crossings` (important for RF signals)
   - `autocorr_lag1` (temporal correlation)

2. **Physics**:
   - `control_settling_time` (control loop analysis)
   - `control_overshoot` (control loop analysis)
   - `phase_dom_freq` (FFT dominant frequency - RF physics)
   - `phase_drift` (rate of phase change)

3. **Precursor**:
   - `var_ratio_late_early` (variance ratio between windows)
   - Per-segment trends (slope/accel for each of 5 segments, not just overall trend)

---

## 6. Architectural Differences

| Aspect | Original | Enhanced | Impact |
|--------|----------|----------|--------|
| **Signal filtering** | 8 primary | All 27 | Enhanced has 3.4x more signals |
| **Array structure** | Numpy arrays | Dict of arrays | Original more efficient |
| **Batch processing** | ❌ No | ✅ Yes | Enhanced better memory |
| **Timeout protection** | ❌ No | ✅ Yes | Enhanced more robust |
| **Resume capability** | ❌ Limited | ✅ Yes | Enhanced more flexible |
| **Modularity** | ✅ 3 separate steps | ❌ Monolithic | Original easier to debug |

---

## 7. Recommendations

### Option A: Keep 27 Signals + Add Missing Features (RECOMMENDED)

**Pros**:
- More comprehensive data (all RF channels)
- Better for exploratory analysis
- Can discover unexpected patterns in secondary channels
- Already tested and working

**Cons**:
- Higher dimensionality (but PCA handles this)
- Some redundancy (but correlation filtering removes it)

**Action Items**:
1. Add missing statistical features: `energy`, `n_zero_crossings`, `autocorr_lag1`
2. Add missing physics features: `control_settling_time`, `control_overshoot`, `phase_dom_freq`
3. Add `var_ratio_late_early` for precursor analysis
4. Keep current 27-signal approach

**Estimated time**: ~30 minutes to add features

---

### Option B: Filter to 8 Primary Signals (Match Original)

**Pros**:
- Exact match with original pipeline
- Focuses on physics-relevant channels
- Lower dimensionality

**Cons**:
- Loses potentially useful information from other channels
- Need to modify signal extraction logic
- Throws away I/Q data that might be valuable

**Action Items**:
1. Add signal filtering in `process_single_file_with_timeout()`
2. Keep only 8 primary SIGNAL_COLS
3. Add all missing features

**Estimated time**: ~45 minutes to refactor + add features

---

### Option C: Hybrid - 8 Primary + 4 I/Q Pairs (12 Total)

**Pros**:
- Balance between original and comprehensive
- Keeps raw I/Q data for advanced analysis
- Still manageable dimensionality

**Cons**:
- Somewhat arbitrary choice
- Partial solution

---

## 8. Feature Count Breakdown

### Original Pipeline (~180 features)
- Statistical: 14 × 8 = 112
- Physics: 16
- Precursor: ~52
  - Trends: 2 signals × 5 segments × 2 (slope+accel) = 20
  - CUSUM: 8 signals = 8
  - Derivatives: 8 signals × 4 = 32
  - Multi-window: 8 signals = 8
- **Total: ~180**

### Enhanced Pipeline (1,157 features)
- Statistical: 14 × 27 = 378
- Physics: 16
- Precursor: ~702
  - Trends: 27 signals × 2 (slope+accel) = 54
  - CUSUM: 27 signals × 2 (max+min) = 54
  - Early/late: 27 signals × 2 (diff+ratio) = 54
  - Derivatives: 27 signals × 6 (vel + accel, each × 3 stats) = 162
  - Segments: 27 signals × 5 segments × 3 (mean+std+slope) = 405
- Metadata: ~10
- **Total: 1,106** (close to observed 1,157)

### After Correlation Filtering
- Original: Unknown (not documented)
- Enhanced: 1,157 → 444 (61% redundancy removed)

### After PCA (95% variance)
- Original: Unknown
- Enhanced: 444 → 24 components

---

## 9. Decision Matrix

| Criterion | Option A (27 signals) | Option B (8 signals) | Option C (12 signals) |
|-----------|----------------------|---------------------|----------------------|
| **Feature completeness** | ⭐⭐⭐ (after fixes) | ⭐⭐⭐ | ⭐⭐ |
| **Match original** | ❌ Different approach | ✅ Exact match | ⚠️ Partial |
| **Data richness** | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| **Efficiency** | ⭐ (more features) | ⭐⭐⭐ | ⭐⭐ |
| **Implementation time** | ⭐⭐⭐ (30 min) | ⭐⭐ (45 min) | ⭐⭐ (40 min) |
| **Downstream ML** | ⭐⭐ (PCA helps) | ⭐⭐⭐ | ⭐⭐⭐ |

---

## 10. Final Recommendation

**RECOMMENDED: Option A - Keep 27 Signals + Add Missing Features**

**Rationale**:
1. **More comprehensive**: Captures all available RF data
2. **Modern ML approach**: Let PCA/correlation filtering handle dimensionality, don't throw away data prematurely
3. **Exploratory advantage**: Can discover patterns in secondary channels
4. **Already validated**: Test run succeeded with 1,157 features
5. **PCA effectiveness**: 1,157 → 24 components proves redundancy is handled well

**Action Plan**:
1. Add 3 missing statistical features (15 min)
2. Add 4 missing physics features (10 min)
3. Add variance ratio feature (5 min)
4. Re-test with 100 files (5 min)
5. Run full pipeline (3-5 hours)

**Total time to deployment**: ~30-35 minutes of coding + testing

---

**Next Steps**: Implement missing features OR proceed with current implementation?
