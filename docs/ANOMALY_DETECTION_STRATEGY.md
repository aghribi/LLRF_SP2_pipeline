# ANOMALY DETECTION STRATEGY FOR SPIRAL2 LLRF DATA

**Project**: Machine Learning for Accelerators - SPIRAL2 LLRF Anomaly Detection
**Date**: December 23, 2025
**Version**: 1.0

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Data Understanding](#data-understanding)
3. [Proposed Pipeline](#proposed-pipeline)
4. [Detailed Methodology](#detailed-methodology)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Expected Outcomes](#expected-outcomes)
7. [References](#references)

---

## Executive Summary

This document proposes a comprehensive, multi-stage anomaly detection pipeline for SPIRAL2 superconducting LINAC Low-Level Radio Frequency (LLRF) data. The strategy combines state-of-the-art techniques from accelerator physics and machine learning, informed by successful deployments at EuXFEL, CEBAF, and other facilities.

### Key Objectives
- **Identify** anomalous patterns in LLRF time-series data indicating potential faults
- **Classify** anomaly types (quenches, breakdowns, RF regulation issues, etc.)
- **Analyze** root causes and correlations with operational parameters
- **Reduce** false alarms while maintaining high detection sensitivity
- **Enable** proactive maintenance and operational optimization

### Approach
The pipeline adopts a **hybrid methodology** combining:
- Domain-specific preprocessing and feature engineering
- Statistical anomaly detection methods
- Unsupervised learning for pattern discovery
- Supervised classification for fault categorization
- Cross-methodology validation for robustness

---

## Data Understanding

### Data Structure

#### Organization
```
data_llrf/
├── data/
│   ├── 2019-2025/        # 7 years of operational data
│   │   ├── CMA/          # Cryomodule A (12 modules)
│   │   │   ├── CMA01-CMA12/  # Each with cavities
│   │   └── CMB/          # Cryomodule B (multiple modules)
│   │       └── CMB01-CMB14/  # Each with 2 cavities
```

#### File Format
- **Binary files** with embedded ASCII headers
- Each file = one **post-mortem event** (fault/alarm trigger)
- Header contains metadata + operational parameters
- Body contains time-series signals (16 channels)

### Key Signals (8 primary channels)

| Signal | Description | Physical Meaning | Units |
|--------|-------------|------------------|-------|
| **Ucav** | Cavity voltage | Main RF amplitude in cavity | kV (calibrated) |
| **PhaseCav** | Cavity phase | RF phase in cavity | degrees |
| **Uci** | Input voltage | Forward RF signal to cavity | kV |
| **PhaseUci** | Input phase | Forward RF phase | degrees |
| **A Ucr** | Control reference amplitude | Setpoint from LLRF controller | kV |
| **A Uamp** | Amplifier voltage | Klystron/amplifier output | kV |
| **courant pickup** | Pickup current | Beam loading indicator | mA |
| **vide** | Vacuum level | Cavity vacuum pressure | mbar |

### Operational Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **NDEC** | Decimation factor | 200 (determines sampling rate) |
| **NROW** | Total samples | Variable (~4000-8000 typical) |
| **POSTROW** | Pre-trigger samples | Number of samples BEFORE alarm |
| **KPI/KII** | Control gains | Proportional/Integral gains |
| **AMPT/PHIT** | Targets | Amplitude/Phase setpoints |
| **LOOP** | Loop status | ON/OFF (closed-loop control) |
| **MASK** | Event mask | 0x007D (CMA) or 0x007F (CMB) |
| **BEAM** | Beam presence | ON/OFF |

### Fault Categories (7 types from ALM field)

1. **Seuil pick-up** - Pickup threshold exceeded
2. **Coupure externe rapide** - Fast external cutoff
3. **Absence autorisation RF** - RF authorization absence
4. **Seuil de vide** - Vacuum threshold exceeded
5. **Claquage ou quench cavité** - Cavity breakdown or quench
6. **Dép seuil de sécurité RF** - RF safety threshold exceeded
7. **Rég signal RF hors tolérance** - RF signal regulation out of tolerance

### Sampling Characteristics
- **Time step**: Δt = (4 / 70.442) × NDEC microseconds
- **For NDEC=200**: Δt ≈ 11.35 µs → **~88 kHz sampling rate**
- **Event duration**: Typically 3000 pre-trigger + 1000 post-trigger samples
- **Total event window**: ~45 ms centered around fault

---

## Proposed Pipeline

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ANOMALY DETECTION PIPELINE                      │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  1. DATA         │  Load binary files, parse headers, extract signals
│  ACQUISITION     │  Filter by quality criteria (KPI≥10, LOOP=ON, etc.)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  2. DATA         │  Signal calibration, noise filtering, normalization
│  PREPROCESSING   │  Temporal alignment, missing value handling
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  3. DATA         │  Statistical analysis, visualization
│  EXPLORATION     │  Feature engineering (domain + statistical)
└────────┬─────────┘  Dimensionality reduction, correlation analysis
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌──────────────────┐              ┌──────────────────┐
│  4a. ANOMALY     │              │  4b. ANOMALY     │
│  DETECTION       │              │  CLASSIFICATION  │
│  (Unsupervised)  │              │  (Supervised)    │
└────────┬─────────┘              └────────┬─────────┘
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
              ┌──────────────────┐
              │  5. ANOMALY      │  Clustering, pattern analysis
              │  ANALYSIS        │  Correlation with parameters
              └────────┬─────────┘  Expert validation
                       │
                       ▼
              ┌──────────────────┐
              │  6. REPORTING &  │  Dashboards, visualizations
              │  VISUALIZATION   │  Automated reports, alerts
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  7. ITERATION    │  Refine models, update features
              │  & IMPROVEMENT   │  Incorporate expert feedback
              └──────────────────┘
```

---

## Detailed Methodology

### 1. Data Acquisition

#### 1.1 Data Loading
**Tool**: PyPostMortem library (existing)

```python
from PyPostMortem.PyPostMortem import Read_Signals

# Load single event
parameters, _, df_signals, df_faults, df_states = Read_Signals(
    filename,
    compute_defauts=True,
    compute_etats=True,
    plot_signaux=False
)
```

**Outputs**:
- `parameters`: Dictionary of operational parameters
- `df_signals`: DataFrame with 8 signal columns
- `df_faults`: Binary flags for 7 fault types
- `df_states`: Binary flags for 16 system states

#### 1.2 Quality Filtering

Apply filters to ensure data quality:

| Filter | Criterion | Rationale |
|--------|-----------|-----------|
| **Control gain** | KPI ≥ 10 | Exclude poorly controlled events |
| **Loop status** | LOOP = "ON" | Only closed-loop operation |
| **Cavity voltage** | mean(Ucav) ≥ 0.1 kV | Exclude very low power events |
| **Input voltage** | mean(Uci) ≥ 0.02 kV | Ensure RF is present |
| **Decimation** | NDEC = 200 | Standardize sampling rate |
| **Fault presence** | At least one fault flag = 1 | Focus on actual events |
| **Mask value** | 0x007D or 0x007F | Standard event masks |

**Expected reduction**: ~40-60% of raw files filtered out

#### 1.3 Dataset Construction

**Metadata DataFrame** (one row per event):
- Filename, cryomodule ID, cavity ID
- Timestamp (year, month, day, hour, minute, second)
- All parameters (NROW, POSTROW, NDEC, KPI, KII, etc.)
- Fault type flags (7 binary columns)
- Mean Ucav, mean Uci (global event characteristics)

**Time-series Arrays** (one array per event):
- Shape: (n_events, n_samples, n_signals)
- Typical: (N, 4000, 8) where N = dataset size
- Aligned to common temporal window relative to trigger

---

### 2. Data Preprocessing

#### 2.1 Signal Calibration

Apply cavity-specific calibration coefficients:

```python
# Cavity type determines coefficients
coef_Ucav = GetCaviteCoef(cavity_type)  # e.g., 9.0 for CAVITE_A
coef_Uci = GetCaviteCoef(cavity_type)   # e.g., 2.7 for CAVITE_A

# Convert raw ADC counts to physical units
Ucav_calibrated = Ucav_raw * coef_Ucav
Uci_calibrated = Uci_raw * coef_Uci
```

**Reference**: PyPostMortem.py GetCaviteCoef() function

#### 2.2 Temporal Alignment

**Challenge**: Events have variable POSTROW (pre-trigger duration)

**Solution**: Standardize temporal window
```python
# Define common window
delta_pre = 3000   # samples before trigger
delta_post = 1000  # samples after trigger

# Extract aligned segment
t0_idx = POSTROW  # Trigger index in original signal
signal_aligned = signal[t0_idx - delta_pre : t0_idx + delta_post]
```

**Result**: All events have same shape (4000, 8) with aligned trigger point

#### 2.3 High-Pass Filtering

**Motivation**: Remove DC offsets and slow drifts

**Method**: Butterworth high-pass filter
```python
from scipy.signal import butter, filtfilt

def highpass_filter(data, cutoff_hz, fs_hz, order=1):
    nyquist = 0.5 * fs_hz
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype='high')
    return filtfilt(b, a, data)

# Apply to each signal
fs = 1 / (4/70.442 * NDEC * 1e-6)  # ~88 kHz for NDEC=200
cutoff = 10  # Hz
signal_filtered = highpass_filter(signal, cutoff, fs, order=1)
```

**Parameters** (from existing main.py):
- Cutoff frequency: 10 Hz
- Filter order: 1 (gentle rolloff)

#### 2.4 Noise Handling

**Strategies**:

1. **Replace zeros**: Avoid log/division errors
   ```python
   signal = signal.replace(0, 1e-10)
   ```

2. **Outlier clipping**: Cap extreme values
   ```python
   # 99.9th percentile clipping per signal
   upper = np.percentile(signal, 99.9)
   signal = np.clip(signal, -upper, upper)
   ```

3. **Smoothing** (optional for visualization):
   ```python
   # Moving average for noisy signals (e.g., vacuum)
   signal_smooth = signal.rolling(window=50, center=True).mean()
   ```

#### 2.5 Normalization

**Method**: Z-score standardization per signal type

```python
# Compute global statistics across all events
mean_Ucav = all_Ucav.mean()
std_Ucav = all_Ucav.std()

# Normalize
Ucav_norm = (Ucav - mean_Ucav) / std_Ucav
```

**Alternative**: Min-max scaling to [0, 1] for bounded signals

**Storage**: Save normalization parameters for inverse transform during interpretation

---

### 3. Data Exploration & Feature Engineering

#### 3.1 Exploratory Data Analysis (EDA)

##### 3.1.1 Temporal Patterns

**Visualizations**:
1. **Time-series plots** of typical vs anomalous events
2. **Phase portraits** (Ucav vs PhaseCav)
3. **Event frequency** over time (daily, weekly, seasonal trends)
4. **Fault type distribution** (bar chart of 7 fault categories)

**Statistical Analysis**:
- Distribution of event durations
- Correlation between pre-trigger behavior and fault type
- Cavity-specific fault rates
- Cryomodule-specific patterns

##### 3.1.2 Signal Correlation Analysis

**Heatmap**: Pearson correlation between 8 signals

Expected strong correlations:
- Uci ↔ Ucav (input drives cavity)
- PhaseUci ↔ PhaseCav (phase relationship)
- A_Ucr ↔ Ucav (control follows setpoint)

**Lag correlation**: Cross-correlation with time shifts
- Reveals control loop delays
- Identifies phase relationships

##### 3.1.3 Parameter Distributions

**Box plots** / histograms for:
- KPI, KII (control parameters)
- AMPT, PHIT (setpoints)
- Mean cavity voltage by fault type
- Vacuum levels (normal vs degraded)

##### 3.1.4 Cavity & Cryomodule Comparison

**Questions to answer**:
- Which cavities have highest fault rates?
- Are CMA and CMB faults similar?
- Do specific modules show degradation over time?

**Visualization**:
- Heatmap grid (cavity × fault type)
- Timeline of faults per module

---

#### 3.2 Feature Engineering

Feature engineering is **critical** for effective anomaly detection. We propose a comprehensive feature set combining domain knowledge and statistical analysis.

##### 3.2.1 Domain-Specific Features

**A. RF Performance Features**

1. **Loaded Quality Factor (Q_L)** - Key indicator from literature
   ```python
   # During decay phase after RF turnoff
   # Exponential fit: V(t) = V0 * exp(-ω_0 * t / (2*Q_L))
   Q_L = fit_exponential_decay(Ucav_decay)
   ```

2. **Cavity Detuning (Δf)**
   ```python
   # From I/Q signals
   I = Ucav * cos(PhaseCav)
   Q = Ucav * sin(PhaseCav)
   detuning = arctan2(Q, I) * f_RF / (2 * pi * Q_L)
   ```

3. **Forward/Reflected Power Balance**
   ```python
   P_forward = Uci^2
   P_reflected = (Uci - Ucav)^2  # Simplified model
   reflection_coefficient = sqrt(P_reflected / P_forward)
   ```

4. **Phase Stability**
   ```python
   phase_jitter = std(PhaseCav)  # Standard deviation
   phase_drift = PhaseCav[-1] - PhaseCav[0]  # Total drift
   ```

5. **Control Error**
   ```python
   error = A_Ucr - Ucav  # Setpoint vs actual
   error_integral = trapz(abs(error), dx=dt)  # Integrated error
   error_peak = max(abs(error))
   ```

**B. Temporal Features**

6. **Pre-trigger vs Post-trigger Statistics**
   ```python
   # Compare behavior before and after alarm
   ratio_mean = mean(signal_post) / mean(signal_pre)
   diff_variance = var(signal_post) - var(signal_pre)
   ```

7. **Fault Propagation Speed**
   ```python
   # Time to drop below threshold
   t_drop = find_first_below_threshold(Ucav, threshold=0.5*Ucav_initial)
   drop_rate = (Ucav_initial - Ucav_min) / t_drop
   ```

8. **Recovery Behavior** (if applicable)
   ```python
   # Does signal recover or continue degrading?
   recovery_slope = linregress(time_post, Ucav_post).slope
   ```

##### 3.2.2 Statistical Features (Time2Feat Library)

Using **tsfresh** or **Time2Feat** library for automated extraction:

**Time-Domain Features** (per signal):
- Mean, median, std, min, max, range
- Skewness, kurtosis (distribution shape)
- Percentiles (25th, 75th, 95th)
- Count above/below mean
- Longest strike above/below mean
- Linear trend slope and intercept

**Derivative Features**:
- First derivative statistics (rate of change)
- Second derivative statistics (acceleration)
- Zero-crossing count

**Autocorrelation Features**:
- ACF at various lags (1, 5, 10, 50, 100)
- Partial autocorrelation function (PACF)

**Frequency-Domain Features**:
- FFT peaks (dominant frequencies)
- Spectral centroid, spread, rolloff
- Spectral entropy
- Welch power spectral density

**Wavelet Features**:
- Discrete wavelet transform coefficients
- Energy in frequency bands

**Complexity Features**:
- Approximate entropy
- Sample entropy
- Hurst exponent (long-range dependence)

**Example Implementation**:
```python
from t2f.extraction.extractor import feature_extraction

# Extract features from array of shape (n_events, n_samples, n_signals)
signal_names = ["A Ucr", "A Uamp", "vide", "courant pickup",
                "Ucav", "PhaseCav", "Uci", "PhaseUci"]

df_features = feature_extraction(
    array_3d,
    batch_size=20,
    p=5,  # AR model order
    signal_names=signal_names,
    compute_intersignal=True  # Cross-signal features
)
# Output: DataFrame with ~2000-5000 features per event
```

##### 3.2.3 Inter-Signal Features

**Cross-correlation features**:
```python
# Correlation between signal pairs
corr_Ucav_Uci = correlate(Ucav, Uci)
max_corr = max(corr_Ucav_Uci)
lag_at_max = argmax(corr_Ucav_Uci) - len(Ucav)  # Time lag
```

**Phase relationships**:
```python
phase_difference = PhaseCav - PhaseUci
phase_diff_mean = mean(phase_difference)
phase_diff_std = std(phase_difference)
```

---

#### 3.3 Feature Selection & Dimensionality Reduction

**Challenge**: 2000-5000 features → curse of dimensionality

##### 3.3.1 Filter Methods

1. **Zero-variance removal**
   ```python
   # Remove features with zero or near-zero variance
   selector = VarianceThreshold(threshold=0.01)
   features_filtered = selector.fit_transform(features)
   ```

2. **Correlation-based filtering**
   ```python
   # Remove highly correlated features (>0.95)
   corr_matrix = features.corr().abs()
   upper_tri = corr_matrix.where(
       np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
   )
   to_drop = [col for col in upper_tri.columns
              if any(upper_tri[col] > 0.95)]
   features_reduced = features.drop(columns=to_drop)
   ```

##### 3.3.2 Wrapper Methods

3. **Recursive Feature Elimination (RFE)**
   ```python
   from sklearn.feature_selection import RFE
   from sklearn.ensemble import RandomForestClassifier

   estimator = RandomForestClassifier(n_estimators=100)
   selector = RFE(estimator, n_features_to_select=100, step=50)
   features_selected = selector.fit_transform(features, labels)
   ```

##### 3.3.3 Embedded Methods

4. **Random Forest Feature Importance**
   ```python
   rf = RandomForestClassifier(n_estimators=200)
   rf.fit(features, labels)
   importances = rf.feature_importances_

   # Keep top N features
   indices = np.argsort(importances)[::-1][:100]
   features_important = features[:, indices]
   ```

5. **L1 Regularization (Lasso)**
   ```python
   from sklearn.linear_model import LassoCV

   lasso = LassoCV(cv=5, random_state=42)
   lasso.fit(features, target)

   # Features with non-zero coefficients
   selected = features[:, lasso.coef_ != 0]
   ```

##### 3.3.4 Dimensionality Reduction

6. **Principal Component Analysis (PCA)**
   ```python
   from sklearn.decomposition import PCA

   # Retain 95% of variance
   pca = PCA(n_components=0.95)
   features_pca = pca.fit_transform(features_normalized)

   # Typically reduces to 50-200 components
   print(f"Reduced to {pca.n_components_} components")
   ```

7. **t-SNE** (for visualization only, not for modeling)
   ```python
   from sklearn.manifold import TSNE

   tsne = TSNE(n_components=2, perplexity=30, random_state=42)
   features_2d = tsne.fit_transform(features_pca)

   # Scatter plot colored by fault type
   plt.scatter(features_2d[:, 0], features_2d[:, 1], c=fault_labels)
   ```

8. **UMAP** (alternative to t-SNE, preserves global structure)
   ```python
   import umap

   reducer = umap.UMAP(n_components=2, random_state=42)
   features_2d = reducer.fit_transform(features_pca)
   ```

##### 3.3.5 Feature Selection Strategy (Recommended)

**Step 1**: Remove low-variance and highly correlated features → ~1000 features

**Step 2**: Apply Random Forest importance → top 200 features

**Step 3**: Apply PCA on selected features → 50-100 components (95% variance)

**Step 4**: Use PCA components for unsupervised learning

**Step 5**: For supervised learning, compare PCA features vs original top 200

---

### 4. Anomaly Detection & Classification

This section describes the **core algorithmic approaches** for detecting and classifying anomalies.

#### 4.1 Unsupervised Anomaly Detection

**Goal**: Discover unusual patterns without labeled data

##### 4.1.1 Statistical Methods

**A. Z-Score Outlier Detection**

```python
# For each feature
z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
outliers = (z_scores > 3).any(axis=1)  # 3-sigma rule
```

**Pros**: Simple, interpretable
**Cons**: Assumes Gaussian distribution, misses multivariate patterns

**B. Mahalanobis Distance**

```python
from scipy.spatial.distance import mahalanobis

# Accounts for feature correlations
cov_matrix = np.cov(X.T)
cov_inv = np.linalg.inv(cov_matrix)
mean_vec = X.mean(axis=0)

distances = [mahalanobis(x, mean_vec, cov_inv) for x in X]
threshold = np.percentile(distances, 95)  # Top 5% as outliers
```

**Pros**: Handles correlated features
**Cons**: Requires invertible covariance (PCA helps)

**C. Isolation Forest**

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.1,  # Expected outlier fraction
    random_state=42
)
predictions = iso_forest.fit_predict(X)  # -1 = outlier, 1 = normal
anomaly_scores = iso_forest.score_samples(X)  # Lower = more anomalous
```

**Pros**: Fast, handles high dimensions, no distance metric needed
**Cons**: Less effective for clustered anomalies

**References**: Liu et al. (2008), successful in LLRF applications

---

##### 4.1.2 Density-Based Clustering

**A. DBSCAN (Density-Based Spatial Clustering)**

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=0.5,  # Neighborhood radius (tune via elbow plot)
    min_samples=10,  # Minimum cluster size
    metric='euclidean'
)
labels = dbscan.fit_predict(X_pca)

# Label -1 indicates noise/outliers
anomalies = (labels == -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
```

**Pros**: Finds arbitrary-shaped clusters, labels outliers
**Cons**: Sensitive to hyperparameters, struggles with varying densities

**B. HDBSCAN (Hierarchical DBSCAN)**

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric='euclidean'
)
labels = clusterer.fit_predict(X_pca)
outlier_scores = clusterer.outlier_scores_  # Continuous anomaly score
```

**Pros**: Handles varying density, more robust
**Cons**: Slower than DBSCAN

**C. OPTICS (Ordering Points To Identify Clustering Structure)**

```python
from sklearn.cluster import OPTICS

optics = OPTICS(
    min_samples=10,
    xi=0.05,
    min_cluster_size=0.05
)
labels = optics.fit_predict(X_pca)
reachability = optics.reachability_  # Density plot
```

**Pros**: Handles variable density clusters
**Cons**: Complex parameter tuning

**Application**: Use clustering to identify normal operational modes, then label data not fitting any cluster as anomalous.

---

##### 4.1.3 Distance-Based Methods

**A. K-Medoids Clustering + Distance Threshold**

```python
from sklearn_extra.cluster import KMedoids

# Cluster normal operational modes
kmedoids = KMedoids(n_clusters=5, metric='euclidean', random_state=42)
kmedoids.fit(X_pca)

# Compute distance to nearest medoid
distances_to_nearest = np.min(
    cdist(X_pca, kmedoids.cluster_centers_), axis=1
)

# Anomaly = far from any cluster center
threshold = np.percentile(distances_to_nearest, 90)
anomalies = distances_to_nearest > threshold
```

**Pros**: Robust medoids (actual data points, not means)
**Cons**: Requires choosing K

**Reference**: Used successfully at EuXFEL with DTW metric

**B. Local Outlier Factor (LOF)**

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1,
    novelty=False  # For training set outliers
)
predictions = lof.fit_predict(X)
outlier_scores = lof.negative_outlier_factor_  # More negative = outlier
```

**Pros**: Detects local anomalies in varying density regions
**Cons**: Computationally expensive for large datasets

---

##### 4.1.4 Reconstruction-Based Methods

**A. Autoencoders (Deep Learning)**

```python
from tensorflow.keras import layers, models

# Encoder-Decoder architecture
input_dim = X.shape[1]
encoding_dim = 32

encoder = layers.Dense(encoding_dim, activation='relu')
decoder = layers.Dense(input_dim, activation='linear')

autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    encoder,
    layers.Dense(16, activation='relu'),  # Bottleneck
    layers.Dense(encoding_dim, activation='relu'),
    decoder
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2)

# Anomaly = high reconstruction error
X_reconstructed = autoencoder.predict(X_test)
reconstruction_errors = np.mean((X_test - X_reconstructed)**2, axis=1)
threshold = np.percentile(reconstruction_errors, 95)
anomalies = reconstruction_errors > threshold
```

**Pros**: Captures complex nonlinear patterns
**Cons**: Requires sufficient data, hyperparameter tuning, GPU for large models

**B. Variational Autoencoder (VAE)**

Similar to autoencoder but learns probabilistic latent space. Anomaly score combines reconstruction error and KL divergence.

**C. PCA Reconstruction Error**

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(X_train)

# Reconstruct
X_reconstructed = pca.inverse_transform(pca.transform(X_test))
reconstruction_errors = np.sum((X_test - X_reconstructed)**2, axis=1)
threshold = np.percentile(reconstruction_errors, 95)
```

**Pros**: Fast, linear baseline
**Cons**: Limited to linear relationships

---

##### 4.1.5 Time-Series Specific Methods

**A. Dynamic Time Warping (DTW) Distance**

```python
from dtaidistance import dtw

# Compare each event to prototype "normal" signal
prototype = median_normal_signal  # From expert-labeled normals

distances = [dtw.distance(event, prototype) for event in events]
threshold = np.percentile(distances, 90)
anomalies = distances > threshold
```

**Pros**: Handles temporal shifts and varying event durations
**Cons**: Computationally expensive O(n²)

**Reference**: Used at EuXFEL for quench detection

**B. Matrix Profile**

```python
import stumpy

# Self-join to find anomalous subsequences
window_size = 100  # Subsequence length
matrix_profile = stumpy.stump(signal, m=window_size)

# High matrix profile value = dissimilar to all other subsequences
anomaly_threshold = np.percentile(matrix_profile[:, 0], 95)
```

**Pros**: Finds anomalous subsequences within longer signals
**Cons**: Requires choosing window size

---

#### 4.2 Supervised Anomaly Classification

**Goal**: Classify events into specific fault types

**Prerequisite**: Labeled dataset (fault type for each event)

##### 4.2.1 Traditional Machine Learning

**A. K-Nearest Neighbors (KNN)**

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',  # Weight by inverse distance
    metric='euclidean'
)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
probabilities = knn.predict_proba(X_test)
```

**Pros**: Simple, no training, interpretable
**Cons**: Slow for large datasets, sensitive to scaling

**B. Random Forest**

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced',  # Handle imbalanced classes
    random_state=42
)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
feature_importances = rf.feature_importances_
```

**Pros**: Robust, handles mixed data types, feature importance
**Cons**: Can overfit, black-box

**C. Gradient Boosting (XGBoost, LightGBM)**

```python
import xgboost as xgb

xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=10,
    scale_pos_weight=ratio_negative_positive,  # Imbalanced data
    random_state=42
)
xgb_clf.fit(X_train, y_train)
```

**Pros**: State-of-the-art performance, handles imbalance
**Cons**: Hyperparameter-heavy, slower training

**D. Support Vector Machine (SVM)**

```python
from sklearn.svm import SVC

svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True  # Enable probability estimates
)
svm.fit(X_train, y_train)
```

**Pros**: Effective in high dimensions
**Cons**: Slow for large datasets, sensitive to scaling

---

##### 4.2.2 Deep Learning Approaches

**A. Multi-Layer Perceptron (MLP)**

```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.005,
    max_iter=200,
    random_state=42
)
mlp.fit(X_train, y_train)
```

**Pros**: Captures nonlinear patterns
**Cons**: Requires more data, hyperparameter tuning

**Reference**: 0.995 AUROC at EuXFEL with NAS-optimized architecture

**B. 1D Convolutional Neural Network (CNN)**

```python
from tensorflow.keras import layers, models

# For raw time-series input (n_samples, n_signals)
model = models.Sequential([
    layers.Input(shape=(4000, 8)),
    layers.Conv1D(32, kernel_size=10, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

**Pros**: Automatically learns features from raw signals
**Cons**: Requires large labeled dataset

**C. LSTM (Long Short-Term Memory)**

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(4000, 8)),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

**Pros**: Captures long-range temporal dependencies
**Cons**: Slow training, prone to overfitting

**Reference**: LSTM-CNN used at CEBAF for fault prediction

**D. Transformer (Attention-Based)**

```python
from tensorflow.keras import layers, models

# Simplified Transformer block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head attention
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)

    # Feed-forward network
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = x + res
    return layers.LayerNormalization(epsilon=1e-6)(x)

# Model
inputs = layers.Input(shape=(4000, 8))
x = transformer_encoder(inputs, head_size=32, num_heads=4, ff_dim=128)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(n_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)
```

**Pros**: State-of-the-art for sequences, parallelizable
**Cons**: Data-hungry, complex

---

##### 4.2.3 Handling Class Imbalance

**Problem**: Fault types may have very different frequencies
- E.g., 80% "quench", 15% "RF regulation", 5% "vacuum"

**Solutions**:

1. **Class weighting**
   ```python
   from sklearn.utils.class_weight import compute_class_weight

   class_weights = compute_class_weight(
       'balanced', classes=np.unique(y_train), y=y_train
   )
   # Pass to model: class_weight={0: w0, 1: w1, ...}
   ```

2. **Resampling**
   ```python
   from imblearn.over_sampling import SMOTE
   from imblearn.under_sampling import RandomUnderSampler

   # Oversample minority classes
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   ```

3. **Stratified splitting**
   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, stratify=y, random_state=42
   )
   ```

---

#### 4.3 Hybrid Approaches (Recommended)

**Strategy**: Combine unsupervised and supervised methods for robustness

##### Approach 1: Two-Stage Pipeline

**Stage 1 (Unsupervised)**: Anomaly detection
- Use Isolation Forest or DBSCAN to flag potential anomalies
- Filters out obvious normal events

**Stage 2 (Supervised)**: Classification
- Apply supervised classifier (RF, XGBoost, MLP) to flagged anomalies
- Classify into specific fault types

**Benefit**: Reduces false positives from supervised model by pre-filtering

##### Approach 2: Ensemble Methods

```python
from sklearn.ensemble import VotingClassifier

# Combine multiple classifiers
rf = RandomForestClassifier(n_estimators=100)
xgb_clf = xgb.XGBClassifier(n_estimators=100)
mlp = MLPClassifier(hidden_layer_sizes=(64, 32))

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb_clf), ('mlp', mlp)],
    voting='soft'  # Weighted average of probabilities
)
ensemble.fit(X_train, y_train)
```

**Benefit**: Robust to individual model weaknesses

##### Approach 3: Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('xgb', xgb.XGBClassifier(n_estimators=100)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# Meta-model
meta_model = LogisticRegression()

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)
stacking.fit(X_train, y_train)
```

**Benefit**: Meta-model learns optimal combination of base predictions

---

#### 4.4 Model Evaluation & Validation

##### 4.4.1 Metrics for Unsupervised Methods

**Clustering Quality**:
```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Silhouette score [-1, 1], higher = better
silhouette = silhouette_score(X, labels)

# Davies-Bouldin index [0, ∞], lower = better
db_index = davies_bouldin_score(X, labels)
```

**Anomaly Detection** (if ground truth available):
```python
from sklearn.metrics import confusion_matrix, classification_report

# Treat as binary: normal (0) vs anomaly (1)
cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))
```

##### 4.4.2 Metrics for Supervised Methods

**Multi-class Classification**:
```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score

# Accuracy (be cautious with imbalanced data)
accuracy = accuracy_score(y_test, y_pred)

# Per-class precision, recall, F1
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')

# AUROC (one-vs-rest for multi-class)
auroc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
```

**Key Metrics** (from accelerator literature):
- **AUROC**: Area Under ROC Curve (target: >0.95)
- **TPR**: True Positive Rate / Recall (target: >95%)
- **FPR**: False Positive Rate (target: <5%)
- **Beam Availability**: Uptime impact (target: minimize false shutdowns)

**Cost-Sensitive Metrics**:
```python
# False negatives (missed faults) are more costly than false positives
# Use F-beta score with β > 1 to emphasize recall
from sklearn.metrics import fbeta_score

f2_score = fbeta_score(y_test, y_pred, beta=2, average='weighted')
```

##### 4.4.3 Cross-Validation Strategies

**K-Fold Cross-Validation**:
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Stratified K-Fold** (for imbalanced data):
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate
```

**Time-Series Split** (if temporal ordering matters):
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # Train on past, test on future
```

##### 4.4.4 Interpretability & Explainability

**Feature Importance** (for tree-based models):
```python
import matplotlib.pyplot as plt

importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:20]  # Top 20

plt.barh(range(20), importances[indices])
plt.yticks(range(20), feature_names[indices])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances')
```

**SHAP (SHapley Additive exPlanations)**:
```python
import shap

# Tree-based models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Individual prediction explanation
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test[0])
```

**LIME (Local Interpretable Model-agnostic Explanations)**:
```python
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    X_train, feature_names=feature_names, class_names=fault_types
)

# Explain single prediction
exp = explainer.explain_instance(X_test[0], model.predict_proba)
exp.show_in_notebook()
```

##### 4.4.5 Uncertainty Quantification

**Prediction Confidence**:
```python
# For probabilistic classifiers
probabilities = model.predict_proba(X_test)
confidence = np.max(probabilities, axis=1)

# Flag low-confidence predictions for expert review
uncertain = confidence < 0.7
```

**Monte Carlo Dropout** (for neural networks):
```python
# Enable dropout during inference
predictions = [model.predict(X_test) for _ in range(100)]
mean_pred = np.mean(predictions, axis=0)
std_pred = np.std(predictions, axis=0)  # Prediction uncertainty
```

**Conformal Prediction**:
```python
from nonconformist.cp import IcpClassifier
from nonconformist.nc import NcFactory

# Wrapper for any sklearn classifier
icp = IcpClassifier(NcFactory.create_nc(model))
icp.fit(X_train, y_train)

# Prediction sets with calibrated confidence
prediction_sets = icp.predict(X_test, significance=0.05)  # 95% confidence
```

---

### 5. Anomaly Analysis

**Goal**: Understand anomalies beyond detection/classification

#### 5.1 Clustering Anomalies

**Question**: Are anomalies of the same type similar?

**Method**: Cluster events within each fault category
```python
# Example: Cluster all "quench" events
quench_events = X[y == 'quench']

kmeans = KMeans(n_clusters=3)
quench_subclusters = kmeans.fit_predict(quench_events)

# Visualize subclusters
tsne = TSNE(n_components=2)
quench_2d = tsne.fit_transform(quench_events)
plt.scatter(quench_2d[:, 0], quench_2d[:, 1], c=quench_subclusters)
plt.title('Quench Event Subclusters')
```

**Insight**: Identify subtypes (e.g., "fast quench" vs "slow quench")

---

#### 5.2 Root Cause Analysis

**Correlation with Parameters**:
```python
# Do certain operational conditions correlate with fault types?
fault_params = metadata[metadata['fault_type'] == 'quench'][['KPI', 'KII', 'AMPT', 'BEAM']]

# Statistical tests (e.g., t-test)
from scipy.stats import ttest_ind

quench_KPI = metadata[metadata['fault_type'] == 'quench']['KPI']
normal_KPI = metadata[metadata['fault_type'] == 'normal']['KPI']

t_stat, p_value = ttest_ind(quench_KPI, normal_KPI)
print(f"KPI difference: t={t_stat:.3f}, p={p_value:.4f}")
```

**Decision Trees for Interpretability**:
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Simple tree to understand fault conditions
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20)
tree.fit(metadata[['KPI', 'KII', 'AMPT', 'mean_Ucav']], metadata['fault_type'])

plot_tree(tree, feature_names=['KPI', 'KII', 'AMPT', 'mean_Ucav'],
          class_names=fault_types, filled=True)
```

**Association Rule Mining**:
```python
from mlxtend.frequent_patterns import apriori, association_rules

# Create binary matrix of conditions
conditions = pd.DataFrame({
    'Low_KPI': metadata['KPI'] < threshold_KPI,
    'High_vacuum': metadata['vide'] > threshold_vide,
    'Beam_ON': metadata['BEAM'] == 'ON',
    'Quench': metadata['fault_type'] == 'quench'
})

frequent_itemsets = apriori(conditions, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(rules[['antecedents', 'consequents', 'confidence']])
```

---

#### 5.3 Temporal Analysis

**Fault Trends Over Time**:
```python
# Aggregate faults by date
metadata['date'] = pd.to_datetime(metadata[['Année', 'Mois', 'Jour']])
fault_timeline = metadata.groupby(['date', 'fault_type']).size().unstack(fill_value=0)

fault_timeline.plot(figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Fault Count')
plt.title('Fault Frequency Over Time')
```

**Insight**: Detect degradation trends, seasonal patterns

**Recurrence Analysis**:
```python
# Time between consecutive faults for same cavity
cavity_faults = metadata[metadata['Cavité'] == 'CAV1'].sort_values('date')
time_between_faults = cavity_faults['date'].diff().dt.total_seconds() / 3600  # Hours

plt.hist(time_between_faults.dropna(), bins=50)
plt.xlabel('Hours Between Faults')
plt.ylabel('Frequency')
```

---

#### 5.4 Cavity & Cryomodule Comparison

**Fault Rate Heatmap**:
```python
# Pivot table: cavity vs fault type
pivot = metadata.pivot_table(
    index='ID', columns='fault_type', aggfunc='size', fill_value=0
)

sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Fault Counts by Cavity')
```

**Survival Analysis**:
```python
# Time-to-failure analysis
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(time_between_faults, event_observed=fault_occurred)
kmf.plot_survival_function()
plt.title('Cavity Survival Curve')
```

---

#### 5.5 Expert Validation

**Process**:
1. **Sample anomalies**: Select diverse examples from each category
2. **Visualization**: Plot time-series signals with annotations
3. **Expert review**: Domain experts label/correct classifications
4. **Feedback loop**: Incorporate corrections into training data

**Tools**:
- Interactive dashboards (Plotly, Bokeh)
- Annotation interfaces (Label Studio, custom Jupyter widgets)

**Example Visualization**:
```python
import plotly.graph_objects as go

# Plot single event with detected anomaly
fig = go.Figure()
for col in ['Ucav', 'PhaseCav', 'Uci']:
    fig.add_trace(go.Scatter(x=time, y=event[col], name=col))

fig.add_vline(x=0, line_dash='dash', line_color='red',
              annotation_text='Trigger')
fig.update_layout(title=f'Event {event_id}: {fault_type}',
                  xaxis_title='Time (µs)', yaxis_title='Signal')
fig.show()
```

---

### 6. Reporting & Visualization

#### 6.1 Dashboards

**Tool**: Plotly Dash, Streamlit, or Grafana

**Components**:

1. **Overview Panel**
   - Total events processed
   - Anomaly detection rate
   - Fault type distribution (pie chart)
   - Model performance metrics

2. **Time-Series Panel**
   - Fault frequency timeline (line/bar chart)
   - Cavity-specific trends
   - Filters: date range, fault type, cavity ID

3. **Signal Viewer**
   - Interactive plots of individual events
   - Multi-signal overlay
   - Zoom, pan, hover tooltips
   - Comparison: normal vs anomalous

4. **Model Performance**
   - Confusion matrix
   - ROC curves
   - Feature importance plot
   - SHAP summary plot

5. **Cavity Health**
   - Heatmap: fault rate by cavity
   - Ranking: most problematic cavities
   - Time-to-failure estimates

**Example (Streamlit)**:
```python
import streamlit as st
import plotly.express as px

st.title('SPIRAL2 LLRF Anomaly Dashboard')

# Sidebar filters
cavity_filter = st.sidebar.multiselect('Cavity', metadata['ID'].unique())
date_range = st.sidebar.date_input('Date Range', [min_date, max_date])

# Filtered data
filtered = metadata[
    (metadata['ID'].isin(cavity_filter)) &
    (metadata['date'].between(*date_range))
]

# Fault distribution
fig = px.pie(filtered, names='fault_type', title='Fault Distribution')
st.plotly_chart(fig)

# Timeline
timeline = filtered.groupby('date').size()
fig = px.line(timeline, title='Faults Over Time')
st.plotly_chart(fig)
```

---

#### 6.2 Automated Reports

**Daily/Weekly Summary Email**:
- Number of new anomalies detected
- Breakdown by fault type
- Top 5 most affected cavities
- Comparison to previous period
- Links to detailed dashboard

**PDF Report Generation**:
```python
from fpdf import FPDF
import matplotlib.pyplot as plt

# Generate plots
plt.figure(figsize=(8, 5))
fault_counts.plot(kind='bar')
plt.savefig('fault_distribution.png')

# Create PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(0, 10, 'Weekly LLRF Anomaly Report', ln=True)
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, f'Period: {start_date} to {end_date}', ln=True)
pdf.image('fault_distribution.png', x=10, y=40, w=180)
pdf.output('weekly_report.pdf')
```

---

#### 6.3 Alert System

**Real-Time Monitoring** (if deployed online):
```python
# Pseudocode for streaming pipeline
while True:
    new_event = fetch_latest_event()

    # Preprocess
    features = extract_features(new_event)

    # Predict
    anomaly_score = model.predict_proba(features)

    # Alert if high-risk
    if anomaly_score > threshold:
        send_alert(
            recipient='operations@ganil.fr',
            subject=f'High-Risk Anomaly: {cavity_id}',
            body=f'Anomaly score: {anomaly_score:.2f}\n'
                 f'Predicted fault: {predicted_fault}\n'
                 f'Timestamp: {timestamp}'
        )
```

**Alert Channels**:
- Email
- SMS (for critical faults)
- Slack/Teams notifications
- EPICS alarm system integration

---

#### 6.4 Visualization Best Practices

1. **Consistent Color Schemes**
   - Use same colors for fault types across all plots
   - Red for critical, yellow for warnings, green for normal

2. **Interactive Plots**
   - Plotly/Bokeh for web dashboards
   - Enable zoom, pan, hover tooltips
   - Linked brushing (selecting in one plot highlights in others)

3. **Context Information**
   - Always include timestamps, cavity IDs
   - Annotate key events (trigger point, detected anomaly)
   - Show confidence intervals/uncertainty

4. **Multi-Scale Views**
   - Overview: system-wide statistics
   - Zoom: individual cavity trends
   - Detail: single event time-series

5. **Comparative Visualization**
   - Side-by-side: normal vs anomalous
   - Before/after: operational changes
   - Cross-cavity: identify patterns

---

### 7. Documentation & Implementation Roadmap

#### 7.1 Documentation Strategy

**Code Documentation**:
- **Docstrings**: NumPy/Google style for all functions
- **Type hints**: Python 3.10+ annotations
- **Comments**: Explain why, not what (code is self-documenting)
- **README.md**: Installation, usage, examples
- **CHANGELOG.md**: Version history

**Methodology Documentation**:
- **Jupyter Notebooks**: Step-by-step analysis with explanations
- **Mathematical Formulas**: LaTeX in Markdown cells
- **References**: Citations to papers, techniques used
- **Rationale**: Explain design choices

**Example Notebook Structure**:
```
1_data_loading.ipynb
├── Overview
├── Data Structure Explanation
├── Code: Load and parse binary files
├── Visualization: Sample signals
└── Summary Statistics

2_preprocessing.ipynb
├── Calibration
│   ├── Formula: Ucav_physical = Ucav_raw × coef
│   └── Code + Validation
├── Filtering
│   ├── High-pass filter design
│   ├── Frequency response plot
│   └── Before/after comparison
└── Normalization

3_feature_engineering.ipynb
├── Domain Features
│   ├── Quality Factor Computation
│   ├── Detuning Calculation
│   └── Control Error Metrics
├── Statistical Features (Time2Feat)
└── Feature Selection
    ├── Correlation heatmap
    ├── RFE results
    └── PCA variance explained

4_unsupervised_detection.ipynb
├── Isolation Forest
├── DBSCAN Clustering
├── Autoencoder
└── Comparison & Ensemble

5_supervised_classification.ipynb
├── Data Splitting (train/test)
├── Handling Imbalance
├── Model Training
│   ├── KNN
│   ├── Random Forest
│   ├── XGBoost
│   └── MLP
├── Hyperparameter Tuning
└── Evaluation

6_model_interpretation.ipynb
├── Feature Importance
├── SHAP Analysis
├── Confusion Matrix Analysis
└── Error Analysis

7_deployment.ipynb
├── Model Saving (joblib/pickle)
├── Batch Prediction Pipeline
├── Dashboard Prototype
└── Future: Real-time Integration
```

---

#### 7.2 Implementation Phases

##### Phase 1: Foundation (Weeks 1-2)
**Deliverables**:
- [ ] Data loading pipeline (PyPostMortem integration)
- [ ] Quality filtering implementation
- [ ] Metadata DataFrame construction
- [ ] Basic EDA (distributions, time-series plots)
- [ ] Notebook: `1_data_loading.ipynb`

**Validation**: Successfully load 1000+ events, verify statistics

---

##### Phase 2: Preprocessing (Weeks 3-4)
**Deliverables**:
- [ ] Signal calibration module
- [ ] Temporal alignment function
- [ ] High-pass filtering pipeline
- [ ] Normalization (fit on training set)
- [ ] Notebook: `2_preprocessing.ipynb`

**Validation**: Compare preprocessed signals to existing work, visual inspection

---

##### Phase 3: Feature Engineering (Weeks 5-7)
**Deliverables**:
- [ ] Domain-specific features (Q_L, detuning, etc.)
- [ ] Time2Feat integration for statistical features
- [ ] Feature selection pipeline (RFE + PCA)
- [ ] Feature importance analysis
- [ ] Notebook: `3_feature_engineering.ipynb`

**Validation**: Check feature distributions, correlations, PCA scree plot

---

##### Phase 4: Unsupervised Detection (Weeks 8-10)
**Deliverables**:
- [ ] Isolation Forest implementation
- [ ] DBSCAN/HDBSCAN clustering
- [ ] Autoencoder (if sufficient data)
- [ ] Ensemble anomaly score
- [ ] Notebook: `4_unsupervised_detection.ipynb`

**Validation**: Visual inspection of flagged anomalies, comparison to known faults

---

##### Phase 5: Supervised Classification (Weeks 11-14)
**Deliverables**:
- [ ] Labeled dataset preparation (expert collaboration)
- [ ] Train/test split (stratified, temporal if applicable)
- [ ] Baseline models (KNN, Random Forest)
- [ ] Advanced models (XGBoost, MLP)
- [ ] Hyperparameter tuning (GridSearch/RandomSearch)
- [ ] Cross-validation
- [ ] Notebook: `5_supervised_classification.ipynb`

**Validation**: AUROC >0.90, F1 >0.85, expert validation on sample

---

##### Phase 6: Interpretation & Analysis (Weeks 15-16)
**Deliverables**:
- [ ] SHAP/LIME explanations
- [ ] Anomaly clustering analysis
- [ ] Root cause correlation study
- [ ] Temporal trend analysis
- [ ] Cavity health ranking
- [ ] Notebook: `6_model_interpretation.ipynb`

**Validation**: Expert review of findings, actionable insights identified

---

##### Phase 7: Deployment & Visualization (Weeks 17-20)
**Deliverables**:
- [ ] Model serialization (joblib)
- [ ] Batch prediction pipeline
- [ ] Interactive dashboard (Streamlit/Dash)
- [ ] Automated report generation
- [ ] User documentation
- [ ] Notebook: `7_deployment.ipynb`

**Validation**: User acceptance testing, dashboard usability

---

##### Phase 8: Iteration & Refinement (Ongoing)
**Activities**:
- [ ] Incorporate expert feedback on classifications
- [ ] Retrain models with corrected labels
- [ ] Add new features based on domain insights
- [ ] Optimize hyperparameters
- [ ] Monitor model performance over time (concept drift)
- [ ] Update documentation

---

#### 7.3 Tools & Environment

**Programming Language**: Python 3.10+

**Core Libraries**:
- **Data**: `pandas`, `numpy`, `h5py`
- **Preprocessing**: `scipy`, `scikit-learn`
- **Feature Engineering**: `tsfresh`, Time2Feat (custom)
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **ML (Traditional)**: `scikit-learn`, `xgboost`, `lightgbm`
- **ML (Deep Learning)**: `tensorflow`, `pytorch` (optional)
- **Clustering**: `hdbscan`, `sklearn_extra` (k-medoids)
- **Interpretability**: `shap`, `lime`
- **Time-Series**: `stumpy` (matrix profile), `dtaidistance` (DTW)
- **Imbalanced Data**: `imbalanced-learn`
- **Reporting**: `fpdf`, `jinja2` (templating)

**Dashboard**: `streamlit` or `plotly-dash`

**Notebooks**: Jupyter Lab

**Version Control**: Git + GitHub/GitLab

**Environment Management**: `conda` or `venv`

**Suggested Environment**:
```yaml
name: llrf_anomaly
channels:
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - seaborn
  - plotly
  - jupyter
  - xgboost
  - lightgbm
  - hdbscan
  - tsfresh
  - shap
  - lime
  - imbalanced-learn
  - streamlit
  - pip:
    - stumpy
    - dtaidistance
```

---

#### 7.4 Computational Resources

**Development**: Laptop/Workstation with 16GB+ RAM

**Training** (if using deep learning):
- GPU: NVIDIA with 8GB+ VRAM (RTX 3060 or better)
- Or cloud: Google Colab Pro, AWS EC2 (g4dn instances)

**Storage**:
- Raw data: ~10-50 GB (depending on years included)
- Processed features: ~1-5 GB
- Models: <100 MB (traditional ML), <1 GB (deep learning)

**Compute Time Estimates**:
- Feature extraction (Time2Feat): ~1-10 hours (10,000 events)
- Model training (Random Forest): ~10-60 minutes
- Deep learning: ~1-5 hours per model (with GPU)
- Hyperparameter search: 10x training time

---

### 8. Expected Outcomes

#### 8.1 Technical Deliverables

1. **Anomaly Detection System**
   - Unsupervised model: Flags anomalies with >90% recall
   - Supervised classifier: Categorizes faults with >85% F1-score
   - Uncertainty quantification for low-confidence predictions

2. **Insights & Analysis**
   - Identification of primary fault patterns
   - Cavity health ranking and degradation trends
   - Operational parameter correlations with faults
   - Recommendations for preventive maintenance

3. **Software Artifacts**
   - Python package for LLRF data processing
   - Jupyter notebooks documenting full pipeline
   - Interactive dashboard for monitoring
   - Automated reporting system

4. **Documentation**
   - Technical report (this document)
   - User guide for dashboard
   - API documentation for code modules
   - Presentation slides for stakeholders

---

#### 8.2 Scientific Contributions

1. **Methodology**
   - Hybrid approach combining domain knowledge and ML
   - Feature engineering strategy for LLRF time-series
   - Benchmark of ML methods on SPIRAL2 data

2. **Publications** (potential)
   - Conference paper: IPAC, LLRF Workshop
   - Journal article: PRAB, NIM-A
   - Internal report: GANIL technical note

3. **Open Science**
   - Code release (GitHub, open-source license)
   - Dataset sharing (anonymized, if permissible)
   - Contributions to Time2Feat library

---

#### 8.3 Operational Impact

1. **Improved Beam Availability**
   - Reduce false alarms → fewer unnecessary shutdowns
   - Early warning of degradation → schedule maintenance
   - Target: +5-10% improvement in uptime

2. **Enhanced Understanding**
   - Quantify fault frequencies and trends
   - Identify systemic vs sporadic issues
   - Inform hardware upgrades (e.g., problematic cavities)

3. **Scalability**
   - Methodology applicable to other accelerators
   - Framework for integrating new fault types
   - Foundation for real-time deployment

---

#### 8.4 Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Insufficient labeled data** | Cannot train supervised models | Start with unsupervised, active learning for labeling |
| **Class imbalance** | Poor minority class performance | SMOTE, class weighting, ensemble methods |
| **Overfitting** | Poor generalization | Cross-validation, regularization, early stopping |
| **Concept drift** | Model degrades over time | Continual learning, periodic retraining |
| **Computational limits** | Cannot process all data | Sampling, cloud resources, optimize code |
| **Expert availability** | Slow validation | Semi-automated tools, clear interfaces |
| **Integration challenges** | Cannot deploy in production | Phased rollout, pilot testing |

---

## References

### Literature (from inputs folder analysis)

1. **EuXFEL Quench Detection**:
   - Parity space method with GLR
   - Neural Architecture Search for lightweight MLPs
   - AUROC: 0.995, TPR: 98.61%, FPR: 0.49%

2. **CEBAF Quality Factor Monitoring**:
   - Q_L decay phase analysis
   - LSTM-CNN for fault prediction
   - Beam availability: >80% (DOE target)

3. **SPIRAL2 (existing work)**:
   - KNN classifier for e-quench detection
   - 100% accuracy on test set
   - Time2Feat feature extraction

4. **Time-Series Analysis**:
   - Dynamic Time Warping (DTW) for pattern matching
   - Matrix Profile for anomaly subsequences
   - Autoencoders for reconstruction-based detection

5. **LLRF Physics**:
   - Loaded quality factor: Q_L = ω_0 / (2 * decay_rate)
   - Detuning: Δf = f_cav - f_RF
   - Reflection coefficient: Γ = sqrt(P_reflected / P_forward)

### Tools & Libraries

- **Time2Feat**: Custom library for time-series feature extraction (SPIRAL2)
- **PyPostMortem**: LLRF binary file reader (SPIRAL2)
- **tsfresh**: Automated time-series feature extraction
- **SHAP**: Model interpretation (Lundberg & Lee, 2017)
- **Imbalanced-learn**: SMOTE and resampling methods
- **HDBSCAN**: Hierarchical density-based clustering

### Standards & Formats

- **HDF5**: Hierarchical Data Format for large datasets
- **EPICS**: Experimental Physics and Industrial Control System
- **FAIR Principles**: Findable, Accessible, Interoperable, Reusable data

---

## Appendix: Glossary

**LLRF**: Low-Level Radio Frequency - Control system for RF cavities
**Ucav**: Cavity voltage amplitude
**Uci**: Input voltage amplitude (forward power)
**Q_L**: Loaded quality factor (resonator efficiency metric)
**NDEC**: Decimation factor (sets sampling rate)
**POSTROW**: Number of pre-trigger samples
**KPI/KII**: Proportional/Integral control gains
**Quench**: Superconducting cavity loses superconductivity
**Breakdown**: Electrical discharge in cavity
**Multipacting**: Resonant electron multiplication
**Field emission**: Electron emission from cavity surface
**Detuning**: Frequency offset from resonance
**GLR**: Generalized Likelihood Ratio (statistical test)
**AUROC**: Area Under Receiver Operating Characteristic curve
**TPR/FPR**: True/False Positive Rate

---

**Document Status**: Draft for Review
**Next Steps**:
1. Review by domain experts (LLRF physicists, accelerator operators)
2. Refinement based on feedback
3. Initiation of Phase 1 implementation

---

*This strategy document is a living document and will be updated as the project progresses.*
