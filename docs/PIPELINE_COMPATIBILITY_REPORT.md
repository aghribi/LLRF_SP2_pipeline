# Pipeline Compatibility Analysis Report
## Steps 04-10 Data Loading Verification

**Date**: 2025-12-27 14:00
**Status**: Task 1.1 COMPLETE
**Analyst**: Claude Code

---

## Executive Summary

✅ **MOSTLY COMPATIBLE** with **2 CRITICAL ISSUES** requiring fixes:

1. **MISSING DATA**: `sequences_pretrigger` array not included in enhanced pipeline output (required by Step 05)
2. **PATH STRUCTURE**: Step 10 expects old directory structure with subdirectories

**Impact**: Steps 04, 06, 07, 08, 09 will work correctly. Steps 05 and 10 require modifications.

---

## Detailed Compatibility Matrix

### What Enhanced Pipeline Outputs

**File**: [`prepare_data_cluster.py`](prepare_data_cluster.py:917-928)
**Output**: `cooked_data/features_engineered.pkl`

```python
{
    'features_all': df_features,           # DataFrame (n_events, n_features)
    'feature_cols': feature_cols_filtered, # List of feature names (after correlation filter)
    'X_scaled': X_scaled,                  # numpy array (n_events, ~500 features)
    'X_pca': X_pca,                        # numpy array (n_events, ~14-20 components)
    'y_binary': y_binary,                  # numpy array (n_events,) - 0=normal, 1=fault
    'y_multilabel': y_multilabel,          # numpy array (n_events, 7) - 7-bit fault vector
    'fault_column_names': [f'Fault_{i}' for i in range(7)],  # Fault labels
    'scaler': scaler,                      # StandardScaler object
    'pca': pca,                            # PCA object
    'metadata': metadata_list,             # List of metadata dicts
}
```

**MISSING (compared to original modular pipeline)**:
```python
{
    'sequences_full': signals_normalized,         # ❌ NOT INCLUDED
    'sequences_pretrigger': pretrigger_signals,   # ❌ NOT INCLUDED (CRITICAL!)
    'sequences_downsampled': signals_downsampled, # ❌ NOT INCLUDED
    'signal_names': signal_names,                 # ❌ NOT INCLUDED
}
```

---

## Step-by-Step Compatibility Analysis

### ✅ Step 04: Anomaly Detection Baseline - COMPATIBLE

**File**: [`prepare_04_anomaly_baseline.py`](cluster_scripts/prepare_04_anomaly_baseline.py:54-68)

**Required Keys**:
- ✅ `X_scaled` - Present
- ✅ `X_pca` - Present
- ✅ `y_binary` - Present (optional, used for evaluation only)

**Verdict**: **FULLY COMPATIBLE** - No changes needed.

---

### ❌ Step 05: Precursor Detection - INCOMPATIBLE

**File**: [`prepare_05_phase0_precursor.py`](cluster_scripts/prepare_05_phase0_precursor.py:54-64)

**Required Keys**:
- ✅ `X_scaled` - Present
- ❌ **`sequences_pretrigger`** - **MISSING!**
- ✅ `y_binary` - Present

**Error at Runtime**:
```python
# Line 64: Will raise KeyError
logger.info(f"  Pre-trigger sequences: {self.feature_data['sequences_pretrigger'].shape}")
# KeyError: 'sequences_pretrigger'
```

**What it is**: Pre-trigger signal array of shape `(n_events, 3000, n_signals)` containing normalized signals for the 3000 samples BEFORE the trigger point.

**Why needed**: For temporal sequence-based precursor detection (RNN/LSTM models).

**Fix Required**: Add `sequences_pretrigger` to enhanced pipeline output (see Fix Section below).

**Verdict**: **INCOMPATIBLE** - Will crash when loading data.

---

### ✅ Step 06: Binary Classification - COMPATIBLE

**File**: [`prepare_06_phase1_binary.py`](cluster_scripts/prepare_06_phase1_binary.py:62-70)

**Required Keys**:
- ✅ `X_scaled` - Present
- ✅ `y_binary` - Present

**Verdict**: **FULLY COMPATIBLE** - No changes needed.

---

### ✅ Step 07: Multi-Label Classification - COMPATIBLE

**File**: [`prepare_07_phase2_multilabel.py`](cluster_scripts/prepare_07_phase2_multilabel.py:34-50)

**Required Keys**:
- ✅ `X_scaled` - Present
- ✅ `y_multilabel` - Present
- ✅ `fault_column_names` - Present

**Verdict**: **FULLY COMPATIBLE** - No changes needed.

---

### ✅ Step 08: Root Cause Identification - COMPATIBLE

**File**: [`prepare_08_phase3a_rootcause.py`](cluster_scripts/prepare_08_phase3a_rootcause.py:34-48)

**Required Keys**:
- ✅ `X_scaled` - Present
- ✅ `y_multilabel` - Present
- ✅ `fault_column_names` - Present

**Verdict**: **FULLY COMPATIBLE** - No changes needed.

---

### ✅ Step 09: Sub-Type Clustering - COMPATIBLE

**File**: [`prepare_09_phase3b_clustering.py`](cluster_scripts/prepare_09_phase3b_clustering.py:32-48)

**Required Keys**:
- ✅ `X_pca` - Present
- ✅ `y_multilabel` - Present
- ✅ `fault_column_names` - Present

**Verdict**: **FULLY COMPATIBLE** - No changes needed.

---

### ⚠️ Step 10: Comprehensive Analysis - PATH ISSUE

**File**: [`prepare_10_comprehensive.py`](cluster_scripts/prepare_10_comprehensive.py:38-63)

**Issue**: Expects old directory structure with subdirectories:

```python
steps = {
    'features': 'step_03_features/features_engineered.pkl',      # ❌ Wrong path
    'anomaly': 'step_04_anomaly/anomaly_baseline.pkl',          # ❌ Wrong path
    'precursor': 'step_05_phase0/precursor_detection.pkl',      # ❌ Wrong path
    'binary': 'step_06_phase1/binary_classification.pkl',       # ❌ Wrong path
    'multilabel': 'step_07_phase2/multilabel_triggers.pkl',     # ❌ Wrong path
    'rootcause': 'step_08_phase3a/root_cause.pkl',              # ❌ Wrong path
    'clustering': 'step_09_phase3b/subtype_clustering.pkl',     # ❌ Wrong path
}
```

**Expected Structure** (modular pipeline):
```
cooked_data/
├── step_03_features/
│   └── features_engineered.pkl
├── step_04_anomaly/
│   └── anomaly_baseline.pkl
├── step_05_phase0/
│   └── precursor_detection.pkl
└── ...
```

**Our Structure** (enhanced pipeline):
```
cooked_data/
├── features_engineered.pkl          # ← At root level
├── anomaly_baseline.pkl             # ← Will be at root level
├── precursor_detection.pkl          # ← Will be at root level
└── ...
```

**Verdict**: **PATH INCOMPATIBLE** - Needs path updates in Step 10 script OR SLURM scripts must create subdirectories.

---

## Critical Issues Summary

### Issue #1: Missing `sequences_pretrigger` (HIGH PRIORITY)

**Affects**: Step 05 (Precursor Detection)

**Impact**: Step 05 will crash with `KeyError: 'sequences_pretrigger'`

**Root Cause**: Enhanced pipeline (`prepare_data_cluster.py`) doesn't store raw signal sequences, only engineered features.

**What is `sequences_pretrigger`**:
- Numpy array of shape `(n_events, 3000, n_signals)`
- Contains normalized signal values for 3000 pre-trigger samples
- Used for temporal/sequence-based modeling (RNN, LSTM, temporal CNNs)

**Source** (from original modular pipeline):
```python
# prepare_03_features.py:423-426
pretrigger_signals = self.segmented_signals.get(
    'full_pretrigger',
    self.signals_normalized[:, :self.config['delta_pre'], :]
)
```

**Solution Options**:

#### Option A: Add Sequences to Enhanced Pipeline Output (RECOMMENDED)
Modify `prepare_data_cluster.py` to include sequence arrays in output.

**Pros**:
- Complete compatibility
- Enables sequence-based models (RNN/LSTM)
- Matches original pipeline exactly

**Cons**:
- Increases output file size significantly (~500MB → ~2-3GB)
- Uses more memory during processing

**Implementation**: See Fix Section below.

---

#### Option B: Modify Step 05 to Work Without Sequences
Remove sequence-based models from Step 05, use only feature-based models.

**Pros**:
- No changes to enhanced pipeline
- Smaller output files

**Cons**:
- Loses temporal modeling capability
- Precursor detection may be less effective
- Diverges from original analysis approach

**Not Recommended** - Reduces pipeline capabilities.

---

### Issue #2: Path Structure Mismatch (MEDIUM PRIORITY)

**Affects**: Step 10 (Comprehensive Analysis), SLURM scripts

**Impact**: Step 10 won't find input files from Steps 03-09

**Solution Options**:

#### Option A: Update SLURM Scripts to Create Subdirectories (RECOMMENDED)
Modify SLURM submission scripts to:
1. Create output subdirectories (step_04_anomaly/, step_05_phase0/, etc.)
2. Update OUTPUT_DIR paths for each step

**Example**:
```bash
# submit_04_anomaly_baseline.sh
OUTPUT_DIR="${COOKED_DATA}/step_04_anomaly"  # Instead of "${COOKED_DATA}"
```

**Pros**:
- No code changes needed
- Matches original structure
- Easier to organize outputs

**Cons**:
- Need to update all SLURM scripts

---

#### Option B: Update Step 10 to Use New Paths
Modify `prepare_10_comprehensive.py` to load from root directory.

**Pros**:
- Minimal changes (1 file)
- Simpler directory structure

**Cons**:
- All result files in same directory (messy)
- Diverges from original structure

---

## Recommended Fixes

### Fix #1: Add Sequences to Enhanced Pipeline (HIGH PRIORITY)

**File to modify**: [`prepare_data_cluster.py`](prepare_data_cluster.py)

**Location**: After line 910 (before saving features_engineered.pkl)

**Code to add**:

```python
# Reconstruct sequence arrays for Steps 05 (precursor detection)
logger.info("Reconstructing sequence arrays for temporal models...")

# Initialize arrays
n_events = len(results)
n_samples = 4000
n_signals = len(all_signal_names)  # 27 signals

sequences_full = np.zeros((n_events, n_samples, n_signals))
sequences_pretrigger = np.zeros((n_events, 3000, n_signals))  # Pre-trigger only
sequences_downsampled = np.zeros((n_events, n_samples // 10, n_signals))

# Fill arrays from results
for i, result in enumerate(results):
    signals_dict = result['signals']
    for j, signal_name in enumerate(all_signal_names):
        if signal_name in signals_dict:
            sig = signals_dict[signal_name]
            sequences_full[i, :, j] = sig
            sequences_pretrigger[i, :, j] = sig[:3000]  # Pre-trigger portion
            sequences_downsampled[i, :, j] = sig[::10]  # Downsample by 10

logger.info(f"Sequence arrays created:")
logger.info(f"  sequences_full: {sequences_full.shape}")
logger.info(f"  sequences_pretrigger: {sequences_pretrigger.shape}")
logger.info(f"  sequences_downsampled: {sequences_downsampled.shape}")

# Add to output dictionary (around line 917)
{
    'features_all': df_features,
    'feature_cols': feature_cols_filtered,
    'X_scaled': X_scaled,
    'X_pca': X_pca,
    'y_binary': y_binary,
    'y_multilabel': y_multilabel,
    'fault_column_names': [f'Fault_{i}' for i in range(7)],
    'scaler': scaler,
    'pca': pca,
    'metadata': metadata_list,

    # ADD THESE:
    'sequences_full': sequences_full,
    'sequences_pretrigger': sequences_pretrigger,
    'sequences_downsampled': sequences_downsampled,
    'signal_names': all_signal_names,
}
```

**Estimated file size increase**: ~500MB → ~2.5GB (5x larger)

**Memory requirement**: Minimal increase (sequences already in memory as part of `results`)

---

### Fix #2: Update SLURM Scripts for Subdirectory Structure (MEDIUM PRIORITY)

**Files to modify**: All `cluster_slurm/submit_*.sh` scripts (Steps 04-10)

**Changes needed**:

```bash
# submit_04_anomaly_baseline.sh
INPUT_DIR="${COOKED_DATA}"                   # Features from root
OUTPUT_DIR="${COOKED_DATA}/step_04_anomaly"  # Results to subdirectory

# submit_05_phase0_precursor.sh
INPUT_DIR="${COOKED_DATA}"
OUTPUT_DIR="${COOKED_DATA}/step_05_phase0"

# submit_06_phase1_binary.sh
INPUT_DIR="${COOKED_DATA}"
OUTPUT_DIR="${COOKED_DATA}/step_06_phase1"

# submit_07_phase2_multilabel.sh
INPUT_DIR="${COOKED_DATA}"
OUTPUT_DIR="${COOKED_DATA}/step_07_phase2"

# submit_08_phase3a_rootcause.sh
INPUT_DIR="${COOKED_DATA}"
OUTPUT_DIR="${COOKED_DATA}/step_08_phase3a"

# submit_09_phase3b_clustering.sh
INPUT_DIR="${COOKED_DATA}"
OUTPUT_DIR="${COOKED_DATA}/step_09_phase3b"

# submit_10_comprehensive.sh
BASE_DIR="${COOKED_DATA}"  # Load from subdirectories
OUTPUT_DIR="${COOKED_DATA}/step_10_comprehensive"
```

**Note**: INPUT_DIR still points to root (where `features_engineered.pkl` is), but OUTPUT_DIR creates subdirectories for each step's results.

---

## Implementation Timeline

### Immediate (Before Job 19146714 Completes)

1. **Fix #1**: Add sequences to enhanced pipeline (~15-20 minutes)
   - Modify `prepare_data_cluster.py`
   - Test with small dataset (100 files)
   - Verify sequence arrays are correct

2. **Fix #2**: Update SLURM scripts (~15 minutes)
   - Update all `submit_*.sh` scripts
   - Verify paths are correct

### After Job Completes

3. Re-run enhanced pipeline with sequence support (if needed)
4. Test Steps 04-10 with updated paths
5. Verify all compatibility issues resolved

---

## Risk Assessment

| Issue | Severity | Impact if Not Fixed | Fix Complexity |
|-------|----------|---------------------|----------------|
| **Missing sequences** | 🔴 HIGH | Step 05 will crash immediately | 🟡 Medium (15 min) |
| **Path structure** | 🟡 MEDIUM | Step 10 won't find inputs | 🟢 Low (15 min) |

---

## Conclusion

**Overall Compatibility**: 6/7 steps fully compatible, 1 step incompatible (Step 05)

**Required Actions**:
1. ✅ **MUST FIX**: Add `sequences_pretrigger` to enhanced pipeline output (Fix #1)
2. ✅ **RECOMMENDED**: Update SLURM scripts for subdirectory structure (Fix #2)

**Timeline**: ~30-35 minutes to implement both fixes

**Next Steps**:
1. Implement Fix #1 (sequences)
2. Test with 100 files
3. Implement Fix #2 (SLURM paths)
4. Proceed to Task 1.4 (create test dataset)

---

**Generated**: 2025-12-27 14:00
**Task**: 1.1 - Verify Steps 04-10 Data Loading Compatibility
**Status**: ✅ COMPLETE
**Next Task**: 1.2 - Implement Fixes (or skip to 1.3 - Update SLURM scripts)
