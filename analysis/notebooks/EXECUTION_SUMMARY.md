# LLRF Anomaly Detection: Notebook Execution Summary

**Date**: 2026-01-10
**Dataset**: features_engineered.pkl (5.7GB, 7,427 events)
**Location**: `/sps/m4cast/_spiral2_data/_llrf_data/cooked_data/`

---

## Executive Summary

Successfully updated and executed three Jupyter notebooks for LLRF anomaly detection data analysis with the new consolidated dataset. All notebooks now work with the updated data structure where features, sequences, and metadata are stored in a single `features_engineered.pkl` file.

### Key Dataset Characteristics
- **Total Events**: 7,427 (4,041 normal, 3,386 fault)
- **Class Balance**: 54.4% normal, 45.6% fault (ratio 1.19:1)
- **Features**: 750 engineered features
- **Signals**: 27 LLRF signals per event
- **Sequence Length**: 4,000 samples (full), 400 samples (downsampled)
- **PCA Components**: 275 (capturing 95% variance)
- **Fault Types**: 7 different fault categories
- **Temporal Coverage**: 2019-09-20 to 2025-10-24
- **Cavities**: 26 unique cavities across 19 cryomodules

---

## Notebook 01: Data Overview Analysis

### Purpose
Comprehensive overview of the LLRF anomaly detection dataset including fault distribution, temporal patterns, and cavity coverage.

### Key Findings

#### 1. Dataset Statistics
- **Total events**: 7,427
- **Normal events (ALM=0)**: 4,041 (54.4%)
- **Fault events (ALM>0)**: 3,386 (45.6%)
- **Class ratio**: 1.19:1 (Normal:Fault) - well-balanced dataset

#### 2. Fault Type Distribution
| Fault Type | Count | Percentage |
|------------|-------|------------|
| Fault_0    | 135   | 1.8%       |
| Fault_1    | 13    | 0.2%       |
| Fault_2    | 2,653 | 35.7%      | ⭐ Dominant fault
| Fault_3    | 128   | 1.7%       |
| Fault_4    | 27    | 0.4%       |
| Fault_5    | 67    | 0.9%       |
| Fault_6    | 638   | 8.6%       |

**Insights**:
- Fault_2 is the dominant fault type (35.7% of all events)
- Fault_1 and Fault_4 are rare (13 and 27 events respectively)
- 92% of fault events have a single fault
- 8% have multiple concurrent faults (up to 3 simultaneous)

#### 3. Multi-label Characteristics
- **Single fault events**: 3,114 (92.0% of faults)
- **Multi-fault events**: 272 (8.0% of faults)
- **Average faults per fault event**: 1.08
- **Max concurrent faults**: 3

**Insight**: The dataset is predominantly single-label, but multi-fault events are present and important for understanding fault cascades.

#### 4. Temporal Distribution
- **Date range**: 2019-09-20 to 2025-10-24 (6+ years)
- **All 7,427 events have valid timestamps**
- Events are distributed across multiple years, providing temporal diversity

#### 5. Cavity Coverage
- **Unique cavities**: 26
- **Unique cryomodules**: 19
- **Top cavities by event count**:
  - CMA07-CAV1: 417 events (5.6%)
  - CMB01-CAV2: 417 events (5.6%)
  - CMA01-CAV1: 413 events (5.6%)

**Insight**: Good coverage across multiple cavities ensures model generalization.

### Outputs Generated
1. `01_fault_distribution.png` - Bar chart of fault type distribution
2. `01_fault_cooccurrence.png` - Co-occurrence matrix and conditional probabilities
3. `01_temporal_distribution.png` - Events by year and monthly timeline
4. `01_data_loading_summary.txt` - Text summary report

---

## Notebook 02: Signal Visualization

### Purpose
Visualize and analyze the 27 LLRF signals to understand normal vs fault signatures and identify discriminative signals.

### Key Findings

#### 1. Signal Structure
- **Events**: 7,427
- **Signals per event**: 27 different LLRF measurements
- **Full sequence length**: 4,000 samples per signal
- **Downsampled sequence**: 400 samples (used for visualization)

#### 2. Signal Types (27 signals)
The 27 signals include various LLRF measurements such as:
- Cavity voltage (Ucav)
- Cavity phase (PhaseCav)
- Forward/reflected power signals
- Vacuum pressure (vide)
- Pickup current (courant pickup)
- Binary control signals
- IQ demodulation signals

#### 3. Most Discriminative Signals
Ranked by mean difference between normal and fault events:

| Rank | Signal | Mean Difference (Δμ) | Significance |
|------|--------|----------------------|--------------|
| 1    | Ucav (Signal 20) | 0.0704 | ⭐⭐⭐ Cavity voltage - primary fault indicator |
| 2    | A Ucr binary (6) | 0.0532 | ⭐⭐ Reflected power |
| 3    | A Uamp binary (7) | 0.0425 | ⭐⭐ Amplifier signal |
| 4    | commandes binary (13) | 0.0423 | ⭐⭐ Control commands |
| 5    | courant pickup (19) | 0.0257 | ⭐ Pickup current - field emission indicator |
| 6    | vide binary (10) | 0.0217 | ⭐ Vacuum pressure |
| 7    | IQ (25) | 0.0139 | IQ demodulation |

**Insights**:
- **Ucav (cavity voltage)** shows the strongest discrimination between normal and fault events
- **Pickup current** and **vacuum pressure** are important for specific fault types
- Control signals and binary indicators also show significant differences

#### 4. Signal Quality
- Signals show appropriate variability (not flat)
- Normal and fault events have visibly different patterns
- Temporal structure is preserved in sequences

### Outputs Generated
1. `02_signal_example_normal.png` - Example of 6 signals from a normal event
2. `02_signal_example_fault.png` - Example of 6 signals from a fault event
3. `02_signal_comparison.png` - Side-by-side comparison of normal vs fault
4. `02_signal_statistics.png` - Overall signal statistics (mean, std, range, CV)
5. `02_signal_statistics_by_class.png` - Signal statistics separated by class

---

## Notebook 03: Comprehensive Feature Exploration

### Purpose
Detailed analysis of the 750 engineered features, including feature taxonomy, PCA dimensionality reduction, and identification of discriminative features per fault type.

### Key Findings

#### 1. Feature Taxonomy (750 total features)

| Category | Count | Description |
|----------|-------|-------------|
| **Statistical - Location** | 306 | Mean, median, min, max |
| **Statistical - Dispersion** | 126 | Std, IQR, range |
| **Temporal - Trends** | 150 | Slopes, accelerations |
| **Temporal - Variance** | 46 | Variance ratios |
| **Statistical - Shape** | 36 | Skewness, kurtosis |
| **Statistical - Correlation** | 19 | Autocorrelation |
| **Change Point** | 16 | CUSUM statistics |
| **Physics-Based** | 16 | Q_L, detuning, control metrics |
| **Statistical - Energy** | 6 | RMS, energy |
| **Statistical - Events** | 3 | Peaks, zero crossings |
| **Total Categorized** | **724** | |
| **Uncategorized** | 26 | |

**Insights**:
- Well-distributed across multiple feature types
- Strong emphasis on statistical characterization (496 features)
- Temporal features (196) capture fault evolution
- Physics-based features provide domain interpretability

#### 2. Feature Quality Statistics

**Variance (after scaling)**:
- Mean variance: 0.987
- Median variance: 1.000 (ideal for scaled features)
- High variance (>1.5): 0 features
- Low variance (<0.1): 10 features (candidates for removal)

**Sparsity**:
- Mean sparsity: 5.3%
- Median sparsity: 0.0%
- Very sparse (>50% zeros): 44 features

**Insight**: Features are well-scaled and non-degenerate. A small number of sparse features exist but most are informative.

#### 3. PCA Dimensionality Reduction

**Results**:
- **Total components extracted**: 275
- **Variance explained**: 95.0%
- **For 90% variance**: 217 components needed
- **For 95% variance**: 275 components needed

**PC1-PC3 Analysis**:
- PC1 explains ~8.5% variance
- PC2 explains ~5.2% variance
- PC3 explains ~3.8% variance
- Classes show visible separation in PC space (especially PC1 vs PC2)

**Insight**: Significant dimensionality reduction achieved (750 → 275 components) while retaining 95% of information. Classes are separable in reduced space.

#### 4. Discriminative Features by Fault Type

##### **Fault_0 (135 events)** - Vacuum-related
Top discriminative features:
1. `vide_velocity_max` (Δ=2.763) - Vacuum pressure rate of change
2. `courant pickup binary_velocity_mean` (Δ=2.347)
3. `courant pickup binary_cusum_max` (Δ=2.228)
4. `vide binary_accel_max` (Δ=2.152)
5. `vide_cusum_max` (Δ=1.876)

**Interpretation**: Characterized by rapid vacuum pressure changes. CUSUM features indicate sustained shifts.

##### **Fault_2 (2,653 events)** - Most common fault
Top discriminative features:
1. `A Ucr binary_q1` (Δ=1.281) - Reflected power quantile
2. `PhaseCav_autocorr_lag1` (Δ=1.278) - Phase temporal correlation
3. `I Ucav binary_q1` (Δ=1.262)
4. `MODP_velocity_std` (Δ=1.197)
5. `IQ_velocity_std` (Δ=1.195)

**Interpretation**: Dominant fault with moderate discriminative power. Involves phase instability and velocity features.

##### **Fault_3 (128 events)** - Control/Vacuum
Top discriminative features:
1. `Uci_median` (Δ=2.035)
2. `vide binary_accel_max` (Δ=2.013)
3. `commandes binary_std` (Δ=1.936)
4. `vide_autocorr_lag1` (Δ=1.675)

**Interpretation**: Related to control command variability and vacuum acceleration.

##### **Fault_4 (27 events)** - Cavity field collapse
Top discriminative features:
1. `Ucav_accel_max` (Δ=4.015) ⭐⭐⭐ **Strongest discriminator**
2. `I Ucav binary_accel_max` (Δ=2.724)
3. `Ucav_velocity_max` (Δ=2.186)
4. `I Ucav binary_velocity_max` (Δ=1.895)

**Interpretation**: Characterized by rapid cavity voltage acceleration - likely quench events. Despite being rare (27 events), features are highly discriminative.

##### **Fault_5 (67 events)** - Decay/frequency
Top discriminative features:
1. `decay_fit_r2` (Δ=1.673) - Quality factor decay fit quality
2. `frequence DSTF binary_velocity_std` (Δ=1.551)
3. `frequence DSTF binary_velocity_max` (Δ=1.517)

**Interpretation**: Related to cavity decay characteristics and frequency detuning.

##### **Fault_6 (638 events)** - Second most common
Top discriminative features:
1. `A Ucr_q1` (Δ=1.102)
2. `I Uci binary_q1` (Δ=1.037)
3. `IQ_min` (Δ=1.036)
4. `Q modulateur binary_q1` (Δ=1.036)

**Interpretation**: Moderate discriminative power. Involves quantile features across multiple signals.

#### 5. Feature Correlation
- Mean absolute correlation: 0.039 (low)
- High correlation (|r|>0.9): 1 pair
- Very high correlation (|r|>0.95): 0 pairs

**Insight**: Features are largely independent, minimal redundancy.

### Outputs Generated
1. `03_feature_distributions.png` - Variance, mean, sparsity, std distributions
2. `03_pca_detailed.png` - 6-panel PCA analysis (scatter plots, variance, histograms)
3. `03_discriminative_features.png` - Top 15 features per fault type with histograms

---

## Technical Improvements Made

### Data Structure Updates
1. **Single file loading**: All data now loaded from `features_engineered.pkl`
2. **Metadata handling**: Converted metadata list to DataFrame automatically
3. **Sequences access**: Full and downsampled sequences now accessible
4. **Backward compatibility**: Scripts work with both old and new data structures

### Execution Strategy
Due to Jupyter/NumPy compatibility issues in the environment, notebooks were executed as Python scripts instead of through Jupyter kernel. This approach:
- Avoided matplotlib/NumPy version conflicts
- Used non-interactive plotting backend (Agg)
- Generated all outputs successfully
- Maintained full functionality

### Files Created
1. `run_notebook_01.py` - Standalone execution script for notebook 01
2. `run_notebook_02.py` - Standalone execution script for notebook 02
3. `run_notebook_03.py` - Standalone execution script for notebook 03
4. `EXECUTION_SUMMARY.md` - This comprehensive summary document

---

## Insights and Recommendations

### Dataset Quality
✅ **Excellent**: Well-balanced, diverse, high-quality dataset
- Good class balance (54% normal, 46% fault)
- Sufficient samples per fault type (except Fault_1 with only 13 events)
- Temporal diversity (6+ years)
- Spatial diversity (26 cavities)

### Feature Engineering
✅ **Comprehensive**: 750 features across 10 categories
- Strong statistical characterization
- Temporal evolution captured
- Physics-based interpretability
- Minimal redundancy

### Dimensionality Reduction
✅ **Effective**: 63% reduction (750 → 275) with 95% variance retained
- Classes separable in PCA space
- Suitable for downstream ML models

### Fault-Specific Insights

1. **Fault_4** (rare but severe): Strongest discriminative features suggest quench events
   - Recommendation: Build specialized detector for this critical fault

2. **Fault_2** (dominant): Moderate discriminative power despite high frequency
   - Recommendation: Multi-class classifier needed, not just binary

3. **Vacuum faults** (Fault_0, Fault_3): Characterized by acceleration features
   - Recommendation: Real-time CUSUM monitoring for early detection

4. **Rare faults** (Fault_1): Only 13 events may not be sufficient for robust modeling
   - Recommendation: Consider data augmentation or collection of more samples

### Next Steps

1. **Model Development**:
   - Train multi-class classifiers using 750 features or 275 PCA components
   - Consider ensemble methods given fault imbalance
   - Implement separate binary classifiers for critical faults (Fault_4)

2. **Temporal Modeling**:
   - Leverage sequence data (4,000 samples) for LSTM/CNN models
   - Explore precursor detection in pre-trigger windows

3. **Feature Selection**:
   - Remove 10 low-variance features
   - Investigate 44 sparse features for relevance
   - Consider fault-specific feature subsets

4. **Validation**:
   - Cross-validation across cavities for generalization
   - Temporal holdout for real-world deployment readiness

---

## Files Generated

### Visualizations (10 images)
1. `01_fault_distribution.png` (39KB)
2. `01_fault_cooccurrence.png` (116KB)
3. `01_temporal_distribution.png` (114KB)
4. `02_signal_example_normal.png` (256KB)
5. `02_signal_example_fault.png` (459KB)
6. `02_signal_comparison.png` (679KB)
7. `02_signal_statistics.png` (105KB)
8. `02_signal_statistics_by_class.png` (64KB)
9. `03_feature_distributions.png` (149KB)
10. `03_pca_detailed.png` (434KB)
11. `03_discriminative_features.png` (775KB)

### Reports
1. `01_data_loading_summary.txt` (994 bytes)

### Scripts
1. `run_notebook_01.py`
2. `run_notebook_02.py`
3. `run_notebook_03.py`

**Total visualization size**: ~3.2 MB

---

## Execution Environment

- **Conda environment**: `/pbs/home/a/aghribi/throng_m4cast/environment/env_llrfspiral2`
- **Python**: 3.10
- **Key libraries**: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn
- **Execution time**: ~2 minutes total across all three notebooks

---

## Conclusion

✅ **Success**: All three notebooks successfully updated and executed with the new 7,427-event dataset. The analysis reveals:

1. **High-quality dataset** suitable for machine learning
2. **Comprehensive feature engineering** with 750 informative features
3. **Fault-specific signatures** identified for targeted detection
4. **Effective dimensionality reduction** enabling efficient modeling

The dataset and analysis provide a solid foundation for developing robust LLRF anomaly detection and fault classification systems for the SPIRAL2 accelerator.

---

**Generated**: 2026-01-10 19:11:00
**By**: Claude Code Assistant
