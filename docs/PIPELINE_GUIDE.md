# LLRF Anomaly Detection Pipeline - Complete Guide

**Version:** 1.0
**Date:** 2025-12-26
**Author:** Claude Code
**Project:** SPIRAL2 LLRF Fault Diagnosis & Precursor Detection

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Pipeline Steps (01-10)](#pipeline-steps)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)
7. [Analysis Workflows](#analysis-workflows)
8. [References](#references)

---

## Overview

### Purpose

This pipeline processes LLRF (Low-Level Radio Frequency) postmortem data from the SPIRAL2 accelerator to:
- **Detect faults** automatically before they cause beam trips
- **Classify fault types** among 7 ALM categories
- **Identify root causes** to prioritize maintenance
- **Discover sub-mechanisms** within fault categories
- **Enable precursor detection** for early warning (>10ms lead time)

### Key Features

✅ **Production-ready**: Cluster-optimized with smart reruns and dependency tracking
✅ **Scalable**: Processes 14,000+ events with batch processing (Step 01: 21h, Steps 02-10: <3h total)
✅ **Multi-phase**: 10 steps from raw data to comprehensive analysis
✅ **Validated**: Corrected trigger classification (ALM-based, not LOOP-based)
✅ **Documented**: Extensive physics context and ML methodology

---

## Architecture

### Directory Structure

```
anomalies/anomalies_exploration/
├── cluster_scripts/           # Processing scripts (Steps 01-10)
│   ├── prepare_01_loading.py
│   ├── prepare_02_preprocess.py
│   ├── prepare_03_features.py
│   ├── prepare_04_anomaly_baseline.py
│   ├── prepare_05_phase0_precursor.py
│   ├── prepare_06_phase1_binary.py
│   ├── prepare_07_phase2_multilabel.py
│   ├── prepare_08_phase3a_rootcause.py
│   ├── prepare_09_phase3b_clustering.py
│   └── prepare_10_comprehensive.py
│
├── cluster_slurm/            # SLURM submission scripts
│   ├── submit_01_loading.sh
│   ├── submit_02_preprocess.sh
│   └── ... (03-10)
│
├── run_pipeline.sh           # Master orchestrator
├── PIPELINE_GUIDE.md         # This file
│
├── cluster_notebooks/        # Original development notebooks
│   └── 01_data_loading_exploration.ipynb, 02_*.ipynb, ...
│
├── cluster_notebooks_analysis/  # Analysis-only notebooks (load cluster outputs)
│   └── Template notebooks for visualizing results
│
└── logs/                     # SLURM job logs
    └── step_XX_JOBID.{out,err}
```

### Data Flow

```
Raw Data (14K files, ~500GB)
    ↓
[Step 01] Data Loading (21h, 128GB RAM)
    → metadata.csv, fault_labels_matrix.npy
    ↓
[Step 02] Preprocessing (15 min, 32GB RAM)
    → preprocessed_data.pkl (~50MB)
    ↓
[Step 03] Feature Engineering (10 min, 16GB RAM)
    → features_engineered.pkl (~12MB)
    ↓
[Steps 04-10] ML Training (<1h total, 8-16GB RAM each)
    → anomaly_baseline.pkl
    → precursor_detection.pkl
    → binary_classification.pkl
    → multilabel_triggers.pkl
    → root_cause.pkl
    → subtype_clustering.pkl
    → comprehensive_analysis.pkl
```

---

## Quick Start

### Prerequisites

- CC-IN2P3 cluster access
- Conda environment: `anomalies`
- SLURM partition: `htc`
- Modules: `Programming_Languages/anaconda/3.11`

### Installation

```bash
# Navigate to project
cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration

# Verify structure
ls -1 cluster_scripts/*.py cluster_slurm/*.sh run_pipeline.sh

# Make executable
chmod +x run_pipeline.sh cluster_slurm/*.sh
```

### Test Run (Dry Run)

```bash
./run_pipeline.sh --dry-run
```

### Run Full Pipeline

```bash
# Run all steps (will take ~24 hours total due to Step 01)
./run_pipeline.sh

# Monitor progress
tail -f logs/step_01_*.out
```

### Resume from Step 03 (if Step 01-02 already complete)

```bash
./run_pipeline.sh --from 03
```

---

## Pipeline Steps

### Step 01: Data Loading

**Script:** `prepare_01_loading.py`
**Time:** ~21 hours (14,203 files)
**Memory:** 128 GB
**CPUs:** 8

**Purpose:**
- Load raw binary files using PyPostMortem
- Extract metadata (NDEC, KPI, ALM, LOOP, timestamps)
- Parse multi-label fault matrix (7 ALM bits → 7 fault types)
- Apply quality filters (NDEC=200, KPI≥10)
- **Critical fix:** Trigger classification based on **ALM field** (not LOOP)

**Inputs:**
- `/sps/m4cast/_spiral2_data/_llrf_data/raw_data/` (14K files)

**Outputs:**
- `step_01_loading/metadata.csv` (~2K events after filtering)
- `step_01_loading/fault_labels_matrix.npy` (N×7 multi-hot encoding)
- `step_01_loading/fault_column_names.pkl`
- `step_01_loading/batches/batch_*.pkl` (intermediate, batch_size=2000)

**Key Corrections Applied:**
```python
# CORRECT (ALM-based):
is_automatic_trigger = (ALM > 0)  # Fault-initiated postmortem
is_manual_acquisition = (ALM == 0)  # Operator-initiated recording

# INCORRECT (old LOOP-based):
is_manual = (LOOP == 'OFF')  # WRONG! LOOP = control loop state
```

**Monitoring:**
```bash
# Check progress
tail -f logs/step_01_*.out

# Check batch completion
ls -1 /sps/m4cast/_spiral2_data/_llrf_data/cooked_data/step_01_loading/batches/ | wc -l
```

---

### Step 02: Signal Preprocessing

**Script:** `prepare_02_preprocess.py`
**Time:** ~10-15 minutes
**Memory:** 32 GB
**CPUs:** 4

**Purpose:**
- Temporal alignment (4000 samples: 3000 pre-trigger + 1000 post-trigger)
- High-pass Butterworth filter (10 Hz, removes DC offset and drift)
- Z-score normalization (zero mean, unit variance)
- Compute temporal derivatives (velocity, acceleration)
- Segment pre-trigger window (5 segments for precursor analysis)

**Signal Processing Pipeline:**
1. **Load raw signals** (8 channels: Ucav, PhaseCav, Uci, PhaseUci, Ucr, Uamp, vide, courant pickup)
2. **Filter:** 10 Hz high-pass → removes Lorentz detuning drift, thermal expansion
3. **Normalize:** Z-score across all events → enables ML convergence
4. **Derivatives:** $\dot{s}(t)$, $\ddot{s}(t)$ → amplifies transient signatures
5. **Segment:** 5 temporal bins in pre-trigger → fault-specific precursor extraction

**Outputs:**
- `step_02_preprocessing/preprocessed_data.pkl` (~50 MB)
  - `signals_normalized`: (N, 4000, 8)
  - `first_derivative_normalized`: (N, 4000, 8)
  - `second_derivative_normalized`: (N, 4000, 8)
  - `segmented_signals`: dict with 7 segments (early, mid_early, mid, mid_late, late, full_pretrigger, post_trigger)
  - `normalization_stats`: mean/std for inverse transform

---

### Step 03: Feature Engineering

**Script:** `prepare_03_features.py`
**Time:** ~5-10 minutes
**Memory:** 16 GB
**CPUs:** 4

**Purpose:**
Extract ~180 features spanning:
1. **Domain-specific physics** (16 features)
2. **Statistical** (112 features = 8 signals × 14 stats)
3. **Precursor signatures** (50+ features)

**Feature Categories:**

| Category | Count | Examples |
|----------|-------|----------|
| **Physics** | 16 | Q_L, detuning, control error, phase stability |
| **Statistical** | 112 | mean, std, skew, kurtosis, peaks, autocorr |
| **Temporal trends** | 20 | Slopes/accelerations in 5 segments (Ucav, PhaseCav) |
| **CUSUM** | 8 | Change point detection per signal |
| **Derivatives** | 32 | Max/mean of 1st & 2nd derivatives |
| **Multi-window** | 8 | Variance ratios (late/early) |
| **TOTAL** | ~180 | → 102 after correlation filtering (>0.95) |
| **PCA** | 18 | 95% variance retention |

**Key Features:**

- **Q_L (Quality Factor):** $Q_L = \pi f_0 \tau_{decay}$ from exponential fit → Detects quench (rapid Q_L drop)
- **Detuning:** $\Delta f = \frac{1}{2\pi} \frac{d\phi}{dt}$ → Lorentz force, microphonics
- **CUSUM:** $S_k = \max(0, S_{k-1} + (x_k - \mu - K))$ → Sustained signal shifts
- **Temporal slopes:** Linear regression per segment → Fault buildup patterns

**Outputs:**
- `step_03_features/features_engineered.pkl` (~12 MB)
  - `features_all`: DataFrame (N × 196 features + metadata)
  - `X_scaled`: (N × 102) StandardScaler output
  - `X_pca`: (N × 18) PCA output
  - `sequences_full`: (N × 4000 × 8) for deep learning
  - `y_binary`: (N,) Fault (1) vs Normal (0)
  - `y_multilabel`: (N × 7) Multi-hot fault labels
  - `scaler`, `pca`: Fitted transformers

---

### Step 04: Anomaly Detection Baseline

**Script:** `prepare_04_anomaly_baseline.py`
**Time:** ~5 minutes
**Memory:** 8 GB
**CPUs:** 4

**Purpose:**
Unsupervised baseline for normal vs anomalous behavior (no labels required).

**Models:**
1. **Isolation Forest:** Random partitioning → Anomalies = easier to isolate
2. **Local Outlier Factor (LOF):** Density-based → Anomalies = low-density regions
3. **DBSCAN:** Clustering → Outliers = points not in any cluster

**Outputs:**
- `step_04_anomaly/anomaly_baseline.pkl`
  - `isolation_forest`: {model, scores, predictions}
  - `lof`: {scores, predictions}
  - `dbscan`: {model, cluster_labels, predictions}

**Use Case:** Detect novel fault types not in training data.

---

### Step 05: Phase 0 - Precursor Detection

**Script:** `prepare_05_phase0_precursor.py`
**Time:** ~5 minutes
**Memory:** 8 GB
**CPUs:** 4

**Purpose:**
Early warning system to predict faults **BEFORE** they occur using pre-trigger features.

**Method:**
- Random Forest on **precursor features** (temporal trends, CUSUM, derivatives)
- Focus on features from **early/mid segments** (T-170ms to T-50ms)
- Goal: >10ms lead time for fault prediction

**Outputs:**
- `step_05_phase0/precursor_detection.pkl`
  - `random_forest`: {model, predictions, probabilities, feature_importance}
  - `cv_scores`: 5-fold cross-validation F1

**Key Metric:** Feature importance → Identifies most predictive precursor signals

---

### Step 06: Phase 1 - Binary Classification

**Script:** `prepare_06_phase1_binary.py`
**Time:** ~10 minutes
**Memory:** 16 GB
**CPUs:** 4

**Purpose:**
Fault vs No-Fault classification (production deployment model).

**Models:**
1. Logistic Regression (baseline, interpretable)
2. Random Forest (non-linear, feature importance)
3. XGBoost (state-of-the-art gradient boosting)
4. SVM (RBF kernel, complex decision boundaries)

**Outputs:**
- `step_06_phase1/binary_classification.pkl`
  - `logistic_regression`, `random_forest`, `xgboost`, `svm`: {model, predictions, probabilities, roc_auc}
  - `data_splits`: Train/test sets for reproducibility

**Production Recommendation:** XGBoost (highest ROC AUC, robust to imbalance)

---

### Step 07: Phase 2 - Multi-Label Classification

**Script:** `prepare_07_phase2_multilabel.py`
**Time:** ~5 minutes
**Memory:** 8 GB
**CPUs:** 4

**Purpose:**
Classify 7 ALM fault types (events can have **multiple simultaneous faults**).

**Fault Types:**
1. Seuil pick-up
2. Coupure externe rapide
3. Absence autorisation RF
4. Seuil de vide
5. Claquage ou quench cavité
6. Dép seuil de sécurité RF
7. Rég signal RF hors tolérance

**Method:**
- Multi-Output Random Forest (independent classifier per fault)
- Hamming Loss metric (proportion of incorrect labels)

**Outputs:**
- `step_07_phase2/multilabel_triggers.pkl`
  - `model`: Multi-output classifier
  - `predictions`: (N × 7) binary matrix
  - `hamming_loss`: Overall error rate

---

### Step 08: Phase 3A - Root Cause Identification

**Script:** `prepare_08_phase3a_rootcause.py`
**Time:** ~5 minutes
**Memory:** 8 GB
**CPUs:** 4

**Purpose:**
Identify **PRIMARY** fault among multiple triggers (root cause vs cascading effects).

**Method:**
- Random Forest on fault-only events
- Target: First active bit in ALM (simplified root cause)
- Physics validation: Temporal ordering, domain knowledge

**Outputs:**
- `step_08_phase3a/root_cause.pkl`
  - `model`: Root cause classifier
  - `predictions`: Primary fault type per event

**Use Case:** Maintenance prioritization (fix root cause, not symptoms)

---

### Step 09: Phase 3B - Sub-Type Clustering

**Script:** `prepare_09_phase3b_clustering.py`
**Time:** ~5 minutes
**Memory:** 8 GB
**CPUs:** 4

**Purpose:**
Discover sub-mechanisms within each fault type (e.g., thermal quench vs field-emission quench).

**Method:**
- K-Means clustering per fault type (k=2-3)
- Silhouette score for cluster quality
- PCA-reduced features for visualization

**Outputs:**
- `step_09_phase3b/subtype_clustering.pkl`
  - Per fault type: {kmeans model, cluster labels, silhouette score}

**Use Case:** Refine fault categorization, identify rare sub-types

---

### Step 10: Comprehensive Analysis

**Script:** `prepare_10_comprehensive.py`
**Time:** ~5 minutes
**Memory:** 8 GB
**CPUs:** 4

**Purpose:**
Aggregate all pipeline outputs and generate comprehensive report.

**Outputs:**
- `step_10_comprehensive/comprehensive_analysis.pkl`
  - All phase results aggregated
  - Cross-phase performance metrics
  - Feature importance rankings
  - Pipeline execution summary

- `step_10_comprehensive/pipeline_report.txt`
  - Human-readable text report
  - Validation status for all phases
  - Final recommendations

---

## Usage Examples

### Example 1: Run Full Pipeline

```bash
cd /pbs/throng/m4cast/projects/SPIRAL2/anomalies/anomalies_exploration

# Check dry run first
./run_pipeline.sh --dry-run

# Run all steps
./run_pipeline.sh

# Total time: ~24 hours (mostly Step 01)
```

### Example 2: Run Only ML Training (Steps 04-10)

Assumes Steps 01-03 already complete:

```bash
./run_pipeline.sh --from 04 --to 10

# Time: <1 hour for all ML phases
```

### Example 3: Rerun Single Step

```bash
# Rerun feature engineering with updated features
./run_pipeline.sh --only 03
```

### Example 4: Run Steps 01-03 Only (Data Preparation)

```bash
./run_pipeline.sh --to 03
```

### Example 5: Check Pipeline Status

```bash
# View help and step list
./run_pipeline.sh --help

# Check which steps are complete
ls -1d /sps/m4cast/_spiral2_data/_llrf_data/cooked_data/step_*/
```

---

## Troubleshooting

### Common Issues

#### 1. Step 01 OOM (Out of Memory)

**Symptom:** Job killed with `Exceeded job memory limit`

**Solution:**
- Reduce `batch_size` in `submit_01_loading.sh`:
  ```bash
  --batch-size 1000  # Instead of 2000
  ```
- Or request more memory:
  ```bash
  #SBATCH --mem=256G  # Instead of 128G
  ```

#### 2. XGBoost Not Found (Step 06)

**Symptom:** `ModuleNotFoundError: No module named 'xgboost'`

**Solution:**
```bash
conda activate anomalies
pip install xgboost
```

#### 3. Step Doesn't Rerun After Fix

**Symptom:** Pipeline skips step even after fixing script

**Solution:**
```bash
# Delete output to force rerun
rm -f /sps/m4cast/_spiral2_data/_llrf_data/cooked_data/step_XX/.../output.pkl

# Or touch input to make it newer
touch /sps/m4cast/_spiral2_data/_llrf_data/cooked_data/step_YY/.../input.pkl
```

#### 4. Missing Fault Labels (Steps 05-10)

**Symptom:** `No binary labels available`

**Solution:**
- Ensure Step 01 completed successfully
- Check `step_01_loading/fault_labels_matrix.npy` exists
- Pass `--step01-dir` in Step 03:
  ```bash
  python cluster_scripts/prepare_03_features.py \
      --input step_02_preprocessing \
      --output step_03_features \
      --step01-dir step_01_loading
  ```

#### 5. SLURM Job Pending Forever

**Symptom:** `squeue` shows job in `PD` (pending) state

**Solution:**
```bash
# Check reason
squeue -u $USER -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %.20R"

# Common reasons:
# - QOSMaxCpuPerUserLimit: Reduce --cpus-per-task
# - QOSMaxMemoryPerUser: Reduce --mem
# - Priority: Wait for higher-priority jobs to finish
```

---

## Analysis Workflows

### Workflow 1: Explore Feature Importance

```python
import pickle

# Load features
with open('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data/step_03_features/features_engineered.pkl', 'rb') as f:
    features = pickle.load(f)

# Load precursor model
with open('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data/step_05_phase0/precursor_detection.pkl', 'rb') as f:
    precursor = pickle.load(f)

# View top precursor features
print(precursor['random_forest']['feature_importance'].head(20))
```

### Workflow 2: Evaluate Binary Classification

```python
import pickle
from sklearn.metrics import classification_report, roc_curve
import matplotlib.pyplot as plt

# Load binary classification results
with open('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data/step_06_phase1/binary_classification.pkl', 'rb') as f:
    binary = pickle.load(f)

# Compare models
for model_name in ['logistic_regression', 'random_forest', 'xgboost', 'svm']:
    roc_auc = binary[model_name]['roc_auc']
    print(f"{model_name}: ROC AUC = {roc_auc:.3f}")

# Plot ROC curve for XGBoost
y_test = binary['data_splits']['y_test']
y_proba = binary['xgboost']['probabilities']

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'XGBoost (AUC={binary["xgboost"]["roc_auc"]:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Binary Classification')
plt.legend()
plt.show()
```

### Workflow 3: Analyze Multi-Label Performance

```python
import pickle
import numpy as np

# Load multi-label results
with open('/sps/m4cast/_spiral2_data/_llrf_data/cooked_data/step_07_phase2/multilabel_triggers.pkl', 'rb') as f:
    multilabel = pickle.load(f)

# Per-label accuracy
y_test = multilabel['data_splits']['y_test']
y_pred = multilabel['predictions']

for i, fault_name in enumerate(multilabel['fault_names']):
    acc = (y_test[:, i] == y_pred[:, i]).mean()
    support = y_test[:, i].sum()
    print(f"{fault_name}: Accuracy={acc:.3f}, Support={support}")
```

---

## References

### Papers & Theses

1. **[PhysRevAccelBeams.26.012801.pdf]** - Parity-space fault detection with GLR signatures
2. **[thppc072.pdf]** - EuXFEL quench prevention with early Q_L monitoring
3. **[2401.15543v1.pdf]** - CEBAF LSTM for precursor anomalies
4. **[fphy-13-1553993.pdf]** - NAS for optimal feature selection
5. **[CharlyLassalle_these_anomalies_18122025.pdf]** - SPIRAL2 anomaly classification context (CRITICAL: ALM vs LOOP clarification)

### Technical Documentation

- **Schilcher, T. (1998)** - "Vector Sum Control of Pulsed Accelerating Fields", DESY Thesis
- **Oppenheim & Schafer (2009)** - "Discrete-Time Signal Processing" (filtering theory)
- **Basseville & Nikiforov (1993)** - "Detection of Abrupt Changes" (CUSUM theory)

### Software

- **PyPostMortem** - SPIRAL2 LLRF data reader (`/pbs/throng/m4cast/projects/SPIRAL2/programmes/PyPostMortem/src`)
- **scikit-learn** - ML algorithms (RF, SVM, PCA, StandardScaler)
- **XGBoost** - Gradient boosting (optional, install via `pip install xgboost`)

---

## Maintenance & Updates

### Adding New Features

1. **Modify** `prepare_03_features.py`:
   - Add feature extraction function
   - Update `process()` to compute new features
   - Add to `precursor_features_dict` or `all_features_list`

2. **Rerun** from Step 03:
   ```bash
   ./run_pipeline.sh --from 03
   ```

### Updating Models

1. **Modify** relevant step script (e.g., `prepare_06_phase1_binary.py`)
2. **Rerun** specific step:
   ```bash
   ./run_pipeline.sh --only 06
   ```

### Adding New Pipeline Step

1. **Create** `cluster_scripts/prepare_11_new_analysis.py`
2. **Create** `cluster_slurm/submit_11_new_analysis.sh`
3. **Update** `run_pipeline.sh`:
   - Increment `END_AT=11`
   - Add Step 11 block
   - Update STEPS array in summary section

---

## Contact & Support

**Project Lead:** Adnan Ghribi (adnan.ghribi@ganil.fr)
**Institution:** GANIL/SPIRAL2
**Pipeline Version:** 1.0 (2025-12-26)

For issues or questions:
1. Check this guide's [Troubleshooting](#troubleshooting) section
2. Review SLURM logs in `logs/step_XX_JOBID.err`
3. Contact project lead with:
   - Step number and job ID
   - Error message from logs
   - Command that triggered the issue

---

**End of Guide**
