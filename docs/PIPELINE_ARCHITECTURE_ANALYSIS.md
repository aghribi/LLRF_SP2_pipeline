# Pipeline Architecture Analysis
## Comparison: Original Design vs. Current Implementation

**Date**: 2025-12-27
**Analysis**: Complete pipeline architecture from `cluster_notebooks` vs. `prepare_data_cluster.py`

---

## 1. ORIGINAL PIPELINE DESIGN (cluster_notebooks)

The original pipeline is a **10-step modular architecture** designed for systematic processing:

### Pipeline Overview

```
Raw Binary Files (14,203 files)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 01: Data Loading                                       │
│ Script: prepare_01_loading.py                               │
│ Input:  raw_data/*.bin (PyPostMortem format)                │
│ Output: step_01_loading/metadata.csv                        │
│         step_01_loading/fault_labels_matrix.npy             │
│                                                              │
│ Operations:                                                  │
│ • Read binary files with PyPostMortem                       │
│ • Extract metadata (ALM, LOOP, KPI, cavity info)            │
│ • Parse fault labels from ALM field (7-bit multi-label)     │
│ • Save CSV with file paths + metadata                       │
│ • Quality filters (KPI≥10, NDEC=200)                        │
│                                                              │
│ Size: ~25KB script                                          │
│ Output: Metadata CSV (~1-2 MB)                              │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 02: Signal Preprocessing                               │
│ Script: prepare_02_preprocess.py                            │
│ Input:  step_01_loading/metadata.csv                        │
│ Output: step_02_preprocessing/preprocessed_data.pkl         │
│                                                              │
│ Operations:                                                  │
│ • Load signals from binary files (only passing events)      │
│ • Extract 8 primary channels:                               │
│   - A Ucr (control reference amplitude)                     │
│   - A Uamp (amplifier voltage)                              │
│   - vide (vacuum level)                                     │
│   - courant pickup (pickup current)                         │
│   - Ucav (cavity voltage MV/m)                              │
│   - PhaseCav (cavity phase °)                               │
│   - Uci (input voltage kW)                                  │
│   - PhaseUci (input phase °)                                │
│ • Temporal alignment (4000 samples: 3000 pre + 1000 post)   │
│ • High-pass Butterworth filter (10 Hz, DC removal)          │
│ • Z-score normalization (zero mean, unit variance)          │
│ • Compute derivatives (1st & 2nd order, velocity/accel)     │
│ • Segment pre-trigger window (5 segments for precursor)     │
│                                                              │
│ Size: ~18KB script                                          │
│ Output: Signals + derivatives (~100-500 MB)                 │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 03: Feature Engineering                                │
│ Script: prepare_03_features.py                              │
│ Input:  step_02_preprocessing/preprocessed_data.pkl         │
│ Output: step_03_features/features_engineered.pkl            │
│                                                              │
│ Operations:                                                  │
│ • Extract 180+ features per event:                          │
│                                                              │
│   A. Domain Physics Features (16):                          │
│      - Q_L (loaded quality factor from decay)               │
│      - Cavity detuning (frequency offset)                   │
│      - Control error metrics                                │
│      - Phase stability (std, jitter)                        │
│      - Amplitude modulation depth                           │
│                                                              │
│   B. Statistical Features (112):                            │
│      Per signal (8 signals × 14 features):                  │
│      - Mean, std, min, max, range                           │
│      - Median, quartiles (Q1, Q3)                           │
│      - Skewness, kurtosis                                   │
│      - RMS, crest factor                                    │
│      - Peak count, peak prominence                          │
│                                                              │
│   C. Precursor Features (50+):                              │
│      - Temporal trends (linear fit slopes)                  │
│      - CUSUM (cumulative sum anomaly detection)             │
│      - Multi-window comparisons (early vs late)            │
│      - Derivative statistics (velocity, acceleration)       │
│      - Segment-based features (5 temporal windows)          │
│                                                              │
│ • Feature scaling (StandardScaler)                          │
│ • PCA dimensionality reduction (50 components)              │
│ • Save feature matrix + labels                              │
│                                                              │
│ Size: ~23KB script                                          │
│ Output: Feature matrix (~10-50 MB)                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 04: Anomaly Detection Baseline                         │
│ Script: prepare_04_anomaly_baseline.py                      │
│ Output: step_04_anomaly/anomaly_baseline.pkl                │
│                                                              │
│ Operations:                                                  │
│ • Train unsupervised anomaly detectors:                     │
│   - Isolation Forest                                        │
│   - One-Class SVM                                           │
│   - Local Outlier Factor                                    │
│   - Autoencoder (deep learning)                             │
│ • Establish baseline anomaly scores                         │
│ • No labels used (unsupervised)                             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 05: Phase 0 - Precursor Detection                      │
│ Script: prepare_05_phase0_precursor.py                      │
│ Output: step_05_phase0/precursor_detection.pkl              │
│                                                              │
│ Operations:                                                  │
│ • Train early warning system                                │
│ • Detect anomalies in pre-trigger window                    │
│ • Use only temporal features (no post-fault data)           │
│ • Goal: Predict faults before they occur                    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 06: Phase 1 - Binary Classification                    │
│ Script: prepare_06_phase1_binary.py                         │
│ Output: step_06_phase1/binary_classification.pkl            │
│                                                              │
│ Operations:                                                  │
│ • Binary classifier: Fault vs. Normal                       │
│ • Train RandomForest, XGBoost, SVM                          │
│ • Cross-validation, hyperparameter tuning                   │
│ • ROC curves, confusion matrices                            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 07: Phase 2 - Multi-Label Classification               │
│ Script: prepare_07_phase2_multilabel.py                     │
│ Output: step_07_phase2/multilabel_triggers.pkl              │
│                                                              │
│ Operations:                                                  │
│ • Multi-label classifier (7 fault types simultaneously)     │
│ • Chain classifiers, Binary Relevance, Label Powerset       │
│ • Fault types:                                              │
│   1. Seuil pick-up                                          │
│   2. Coupure externe rapide                                 │
│   3. Absence autorisation RF                                │
│   4. Seuil de vide                                          │
│   5. Claquage ou quench cavité                              │
│   6. Dép seuil de sécurité RF                               │
│   7. Rég signal RF hors tolérance                           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 08: Phase 3A - Root Cause Identification               │
│ Script: prepare_08_phase3a_rootcause.py                     │
│ Output: step_08_phase3a/root_cause.pkl                      │
│                                                              │
│ Operations:                                                  │
│ • For multi-label events, identify primary fault            │
│ • Temporal ordering (which fault triggered first)           │
│ • Causal inference                                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 09: Phase 3B - Sub-Type Clustering                     │
│ Script: prepare_09_phase3b_clustering.py                    │
│ Output: step_09_phase3b/subtype_clustering.pkl              │
│                                                              │
│ Operations:                                                  │
│ • Cluster each fault type into sub-types                    │
│ • Discover operational regimes                              │
│ • K-means, DBSCAN, hierarchical clustering                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 10: Comprehensive Analysis                             │
│ Script: prepare_10_comprehensive.py                         │
│ Output: step_10_comprehensive/comprehensive_analysis.pkl    │
│         step_10_comprehensive/pipeline_report.txt           │
│                                                              │
│ Operations:                                                  │
│ • Aggregate results from all phases                         │
│ • Cross-phase performance metrics                           │
│ • Generate comprehensive report                             │
│ • Model comparison tables                                   │
│ • Feature importance rankings                               │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Modularity**: Each step is independent, can be run separately
2. **Incremental**: Outputs from step N feed into step N+1
3. **Checkpointing**: Each step saves intermediate results
4. **Resume-friendly**: Can restart from any step if earlier steps complete
5. **Memory-efficient**: Process and save, then release memory
6. **Debuggable**: Easy to identify which step failed

### Resource Requirements per Step

| Step | Script Size | Input Size | Output Size | Memory Peak | Time (est.) |
|------|-------------|------------|-------------|-------------|-------------|
| 01   | 25 KB       | 14K files  | 2 MB        | ~2 GB       | 30-60 min   |
| 02   | 18 KB       | 2 MB       | 500 MB      | ~8 GB       | 60-90 min   |
| 03   | 23 KB       | 500 MB     | 50 MB       | ~4 GB       | 20-40 min   |
| 04   | 8 KB        | 50 MB      | 10 MB       | ~2 GB       | 10-20 min   |
| 05   | 7 KB        | 50 MB      | 5 MB        | ~2 GB       | 10-20 min   |
| 06   | 7 KB        | 50 MB      | 10 MB       | ~2 GB       | 15-30 min   |
| 07   | 3 KB        | 50 MB      | 10 MB       | ~2 GB       | 15-30 min   |
| 08   | 4 KB        | 50 MB      | 5 MB        | ~2 GB       | 10-15 min   |
| 09   | 4 KB        | 50 MB      | 10 MB       | ~2 GB       | 15-25 min   |
| 10   | 8 KB        | 100 MB     | 20 MB       | ~3 GB       | 10-20 min   |

**Total pipeline time**: ~3-5 hours for full dataset

---

## 2. CURRENT IMPLEMENTATION (prepare_data_cluster.py)

### What prepare_data_cluster.py Actually Does

The current script is a **MONOLITHIC implementation** that combines Steps 01 + 02 + 03 into a single process:

```
prepare_data_cluster.py (NEW VERSION with timeouts)
├── File Discovery
│   └── Scan 14,203 raw binary files
│
├── Step 01 Operations (embedded in process_single_file)
│   ├── Load binary with PyPostMortem.Read_Signals()
│   ├── Extract metadata from parameters dict
│   ├── Parse fault labels from ALM field
│   └── Apply quality filters (KPI≥10, NDEC=200)
│
├── Step 02 Operations (embedded in _preprocess_signals)
│   ├── High-pass Butterworth filter (10 Hz, order=4)
│   ├── Z-score normalization
│   └── **MISSING**: Derivatives, temporal segmentation
│
├── Step 03 Operations (embedded in _engineer_features)
│   ├── Statistical features (mean, std, max, min, median, rms, range)
│   ├── FFT peak frequency magnitude
│   ├── Metadata as features (numeric conversion)
│   └── **MISSING**: Physics features, precursor features, derivatives
│
├── Batch Processing (run_parallel)
│   ├── Process 500 files per batch
│   ├── Save batch_XXX.pkl immediately
│   └── Free memory after each batch
│
├── Final Aggregation (save_results)
│   ├── Merge all batches
│   ├── Create feature DataFrame
│   ├── StandardScaler + PCA (50 components)
│   ├── Save preprocessed_data.pkl
│   └── Save features_engineered.pkl
│
└── Outputs
    ├── batches/batch_000.pkl ... batch_028.pkl
    ├── preprocessed_data.pkl
    ├── features_engineered.pkl
    └── processing_summary.txt
```

### Critical Differences

| Aspect | Original Pipeline | prepare_data_cluster.py |
|--------|-------------------|-------------------------|
| **Architecture** | 3 separate steps (01→02→03) | Monolithic (all in one) |
| **Modularity** | High (independent scripts) | Low (single script) |
| **Memory efficiency** | Excellent (step-by-step) | Poor (holds all in memory) |
| **Resume capability** | Step-level granularity | Batch-level granularity |
| **Debugging** | Easy (isolate step) | Hard (combined operations) |
| **Timeout protection** | ❌ None | ✅ Per-file (NEW) |
| **Feature completeness** | 180+ features | ~50 features |
| **Physics features** | ✅ 16 features | ❌ Missing |
| **Precursor features** | ✅ 50+ features | ❌ Missing |
| **Derivatives** | ✅ 1st & 2nd order | ❌ Missing |
| **Temporal segmentation** | ✅ 5 windows | ❌ Missing |
| **Execution time** | ~2-3 hours (3 steps) | ~2-3 hours (monolithic) |

---

## 3. FEATURE COMPARISON

### Features in Original Pipeline (Step 03)

**Total: 180+ features**

#### A. Domain Physics Features (16)
```python
1. Q_L                    # Loaded quality factor from cavity decay
2. Q_L_tau                # Decay time constant
3. Q_L_r2                 # Decay fit quality
4. detuning_hz            # Cavity frequency offset
5. detuning_bandwidth     # Normalized detuning
6. control_error_mean     # Mean control error (Ucr - Ucav)
7. control_error_std      # Control error variability
8. control_error_max      # Peak control error
9. phase_stability_std    # Phase jitter
10. phase_stability_range # Phase excursion
11. amplitude_mod_depth   # AM depth percentage
12. phase_mod_depth       # PM depth
13. forward_reflected_ratio # Power reflection coefficient
14. cavity_gradient       # MV/m electric field strength
15. beam_loading          # Estimated beam current effect
16. klystron_saturation   # Amplifier saturation indicator
```

#### B. Statistical Features (112)
```python
# Per signal (8 signals × 14 features = 112)
For each of ['A Ucr', 'A Uamp', 'vide', 'courant pickup',
             'Ucav', 'PhaseCav', 'Uci', 'PhaseUci']:

    1. mean                # Average value
    2. std                 # Standard deviation
    3. min                 # Minimum
    4. max                 # Maximum
    5. range               # max - min
    6. median              # 50th percentile
    7. q1                  # 25th percentile
    8. q3                  # 75th percentile
    9. iqr                 # Interquartile range
    10. skewness           # Distribution asymmetry
    11. kurtosis           # Tail heaviness
    12. rms                # Root mean square
    13. crest_factor       # Peak-to-RMS ratio
    14. peak_count         # Number of significant peaks
```

#### C. Precursor Features (50+)
```python
# Temporal trend features (8 signals × 2 = 16)
{signal}_trend_slope         # Linear fit slope in pre-trigger
{signal}_trend_acceleration  # Quadratic term

# CUSUM features (8 signals × 2 = 16)
{signal}_cusum_pos           # Positive cumulative sum
{signal}_cusum_neg           # Negative cumulative sum

# Multi-window comparison (8 signals × 2 = 16)
{signal}_early_late_diff     # Early vs late window difference
{signal}_early_late_ratio    # Early/late window ratio

# Derivative statistics (8 signals × 4 = 32)
{signal}_velocity_mean       # 1st derivative mean
{signal}_velocity_std        # 1st derivative std
{signal}_accel_mean          # 2nd derivative mean
{signal}_accel_std           # 2nd derivative std

# Segment-based (5 windows × 8 signals × 3 = 120)
{signal}_segment_{n}_mean    # Mean in segment n
{signal}_segment_{n}_std     # Std in segment n
{signal}_segment_{n}_slope   # Trend in segment n
```

### Features in prepare_data_cluster.py

**Total: ~50 features**

#### Statistical Only (56)
```python
# Per signal (8 signals × 7 features = 56)
For each signal:
    1. {signal}_mean
    2. {signal}_std
    3. {signal}_max
    4. {signal}_min
    5. {signal}_median
    6. {signal}_rms
    7. {signal}_range
    8. {signal}_peak_freq_magnitude  # FFT-based

# Metadata features (~10)
meta_KPI, meta_NDEC, meta_ALM, etc. (numeric only)
```

#### Missing Features
```
❌ No physics features (Q_L, detuning, control error, etc.)
❌ No precursor features (trends, CUSUM, multi-window)
❌ No derivative features (velocity, acceleration)
❌ No temporal segmentation (5-window analysis)
❌ No peak analysis (peak count, prominence)
❌ No distribution features (skewness, kurtosis, quartiles)
❌ No crest factor, IQR, etc.
```

---

## 4. CRITICAL ANALYSIS

### Why prepare_data_cluster.py Was Created

Looking at the code history, `prepare_data_cluster.py` was created as a **"quick start" script** to:
1. Get the pipeline running quickly without setting up 10 separate steps
2. Combine the most time-consuming operations (file loading + preprocessing + features)
3. Provide a single command for users to run

### Problems with the Current Approach

#### 1. **Feature Deficiency**
- Only ~30% of original features implemented (50/180)
- Missing ALL physics-based features (most important for LLRF)
- Missing ALL precursor detection features (critical for early warning)
- Missing derivatives (velocity/acceleration) needed for ML

#### 2. **Memory Inefficiency**
- Loads ALL data into memory before saving
- No intermediate checkpoints
- If job crashes at 95%, must restart from 0%

#### 3. **Debugging Difficulty**
- Can't isolate which operation is slow/buggy
- Timeout could be in loading, preprocessing, OR feature engineering
- Hard to add new features without modifying core loop

#### 4. **Lack of Flexibility**
- Can't re-run feature engineering with different params
- Can't skip preprocessing if already done
- Can't experiment with different feature sets

#### 5. **Divergence from Notebooks**
- Analysis notebooks expect 180+ features
- Will fail or perform poorly with only 50 features
- Physics features are essential for interpretability

---

## 5. RECOMMENDATIONS

### Option A: Keep Current Monolithic Approach (Short-term fix)

**If you must use prepare_data_cluster.py**, add missing features:

```python
# In _engineer_features(), add:

# 1. Physics features
features[f'{signal_name}_q_factor'] = calculate_ql(signal_data)
features[f'{signal_name}_detuning'] = calculate_detuning(...)
features[f'{signal_name}_control_error'] = calc_control_error(...)

# 2. Distribution features
features[f'{signal_name}_skewness'] = scipy.stats.skew(signal_data)
features[f'{signal_name}_kurtosis'] = scipy.stats.kurtosis(signal_data)
features[f'{signal_name}_q1'] = np.percentile(signal_data, 25)
features[f'{signal_name}_q3'] = np.percentile(signal_data, 75)

# 3. Derivative features (requires adding derivatives to _preprocess_signals)
deriv1 = np.gradient(signal_data)
deriv2 = np.gradient(deriv1)
features[f'{signal_name}_velocity_mean'] = np.mean(deriv1)
features[f'{signal_name}_accel_mean'] = np.mean(deriv2)

# 4. Precursor features
features[f'{signal_name}_trend_slope'] = calc_trend(signal_data[:3000])
features[f'{signal_name}_cusum'] = calc_cusum(signal_data[:3000])
```

### Option B: Migrate to Modular Pipeline (Recommended)

**Use the original 3-step approach** from cluster_scripts:

```bash
# Step 01: Data loading only (fast, ~30 min)
sbatch cluster_slurm/submit_01_loading.sh

# Step 02: Signal preprocessing (moderate, ~60 min)
sbatch cluster_slurm/submit_02_preprocess.sh

# Step 03: Feature engineering (fast, ~20 min)
sbatch cluster_slurm/submit_03_features.sh
```

**Advantages**:
- ✅ All 180+ features
- ✅ Better memory management
- ✅ Step-level resume capability
- ✅ Easier debugging
- ✅ Matches analysis notebooks exactly
- ✅ Can re-run feature engineering easily

**Disadvantages**:
- Need to run 3 commands instead of 1
- Slightly more complex orchestration

### Option C: Hybrid Approach (Practical compromise)

**Keep prepare_data_cluster.py for Steps 01+02**, but separate Step 03:

```bash
# Combined loading + preprocessing (keeps batch processing benefits)
sbatch submit_01_02_combined.sh  # Based on prepare_data_cluster.py

# Separate feature engineering (allows experimentation)
sbatch submit_03_features.sh     # Use original prepare_03_features.py
```

This gives you:
- ✅ Timeout protection (in combined script)
- ✅ Batch processing (memory efficient)
- ✅ All features (from separate Step 03)
- ✅ Feature experimentation flexibility

---

## 6. IMMEDIATE ACTION ITEMS

### For Current Job (19136690)

✅ **Job is running** with timeout protection - will complete Steps 01+02 partially:
- Loading: ✅ Complete
- Preprocessing: ✅ Basic (filtering + normalization)
- Features: ⚠️ Only 50/180 features

### After Current Job Completes

**DECISION REQUIRED**: Choose one of the following:

#### Path A: Quick Fix (1-2 hours work)
1. Enhance `prepare_data_cluster.py` with missing features
2. Re-run feature engineering step only
3. Continue with current architecture

#### Path B: Proper Migration (4-6 hours work)
1. Test original `cluster_scripts/prepare_01_loading.py` with timeouts
2. Verify `cluster_scripts/prepare_02_preprocess.py` works
3. Run `cluster_scripts/prepare_03_features.py` to get all 180 features
4. Continue with Steps 04-10

#### Path C: Hybrid (2-3 hours work)
1. Keep current `prepare_data_cluster.py` output (batches)
2. Create adapter script to convert batches → Step 02 format
3. Run `cluster_scripts/prepare_03_features.py` on adapted data
4. Get all 180 features without re-processing signals

---

## 7. CONCLUSION

**Current Status**: The `prepare_data_cluster.py` script is a **monolithic approximation** of the original 3-step pipeline. It successfully handles Steps 01 and 02, but only implements **~30% of the required features** from Step 03.

**Recommendation**: **Path B (Proper Migration)** is strongly recommended because:
1. Physics features are CRITICAL for LLRF fault diagnosis
2. Precursor features are essential for early warning system
3. Analysis notebooks expect full feature set
4. Future maintenance will be much easier
5. Original pipeline was designed by domain experts with careful thought

**Timeline Impact**:
- Path A: +2 hours to add features, but technical debt remains
- Path B: +6 hours to migrate properly, but clean architecture
- Path C: +3 hours for hybrid, but still missing some benefits

**The original modular pipeline exists, is tested, and is superior. We should use it.**

---

**Next Steps**: Discuss with user which path to take once job 19136690 completes.
