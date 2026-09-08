# Feature Engineering Reference Guide

Quick reference for the 1,243 extracted features from LLRF post-mortem signals.

## Feature Categories

| Category | Features | Description |
|----------|----------|-------------|
| **Physics-Based** | ~16 | Domain-specific LLRF/cavity metrics |
| **Statistical** | ~900 | Moments, energy, events across signals |
| **Temporal Precursor** | ~250 | Trends, slopes, variance ratios |
| **Change Point** | ~8 | CUSUM statistics |
| **Rate-of-Change** | ~60 | First/second derivatives |
| **TOTAL** | **~1,243** | |

---

## 1. Physics-Based Features (16 features)

### Quality Factor ($Q_L$)
```
ql                  # Quality factor (dimensionless)
decay_tau           # Decay time constant (seconds)
ql_fit_r2           # Fit quality (R²)
```

**Formula**: $Q_L = \pi f_0 \tau$ where $f_0 = 88$ MHz

**Physical Range**:
- Normal: $Q_L \sim 10^9$
- Quench: $Q_L \sim 10^6$ - $10^7$

---

### Detuning
```
detuning_hz         # Frequency offset from resonance (Hz)
phase_drift         # Phase drift rate (rad/s)
```

**Formula**: $\Delta f = \frac{1}{2\pi} \frac{d\phi}{dt}$

**Typical Values**: -100 Hz to +100 Hz (Lorentz force, microphonics)

---

### Control Loop Performance
```
control_rms_pre        # RMS error before trigger
control_rms_post       # RMS error after trigger
control_max_error      # Peak error
control_settling_time  # Time to settle (seconds)
control_overshoot      # Maximum overshoot
```

---

### Phase Stability
```
phase_jitter_cav       # Cavity phase std (radians)
phase_excursion_cav    # Cavity phase peak-to-peak
phase_dom_freq_cav     # Dominant frequency in cavity phase
phase_jitter_uci       # Uci phase std
phase_excursion_uci    # Uci phase peak-to-peak
phase_dom_freq_uci     # Dominant frequency in Uci phase
```

---

## 2. Statistical Features (~900 features)

**Applied to 8 signals**: Ucav, PhaseCav, Uci, PhaseUci, A Ucr, A Uamp, vide, courant pickup

**Across multiple windows**: full, pre-trigger, post-trigger, early, mid, late

### Feature Template (per signal, per window)

```python
{signal}_mean              # Mean value
{signal}_std               # Standard deviation
{signal}_min               # Minimum value
{signal}_max               # Maximum value
{signal}_median            # Median value
{signal}_iqr               # Interquartile range
{signal}_range             # max - min
{signal}_skewness          # Distribution asymmetry
{signal}_kurtosis          # Distribution tail weight
{signal}_energy            # Sum of squares
{signal}_rms               # Root mean square
{signal}_n_peaks           # Number of peaks
{signal}_n_zero_crossings  # Zero crossing count
{signal}_autocorr_lag1     # Lag-1 autocorrelation
```

### Example Features
```
Ucav_full_mean
Ucav_pretrigger_std
PhaseCav_early_max
vide_late_energy
courant_pickup_rms
```

---

## 3. Temporal Precursor Features (~250 features)

### Linear Trends (Slopes)
**Formula**: $\beta = \frac{\sum(t - \bar{t})(y - \bar{y})}{\sum(t - \bar{t})^2}$

**Segments**: early, mid_early, mid, mid_late, late

```
Ucav_early_slope           # Slope in early segment
Ucav_mid_slope             # Slope in mid segment
Ucav_late_slope            # Slope in late segment (quench indicator)
PhaseCav_early_slope       # Phase trend early
PhaseCav_late_slope        # Phase trend late
```

**Physical Interpretation**:
- Negative slope in $U_{cav}$ late → Quench precursor
- Positive slope in $I_{pickup}$ → Field emission buildup

---

### Quadratic Trends (Accelerations)
**Formula**: Fit $y = a_0 + a_1 t + a_2 t^2$, extract $a_2$

```
Ucav_early_accel
Ucav_mid_accel
Ucav_late_accel
PhaseCav_early_accel
PhaseCav_late_accel
```

**Physical Interpretation**:
- $a_2 > 0$ → Accelerating growth (unstable)
- $a_2 < 0$ → Accelerating decay (collapse)

---

### Multi-Window Variance Ratio
**Formula**: $R = \frac{\sigma^2_{late}}{\sigma^2_{early}}$

```
Ucav_var_ratio_late_early
PhaseCav_var_ratio_late_early
Uci_var_ratio_late_early
vide_var_ratio_late_early
courant_pickup_var_ratio_late_early
```

**Interpretation**:
- $R \gg 1$ → Growing instability (field emission)
- $R \approx 1$ → Stable
- $R < 1$ → Damping

---

## 4. Change Point Detection (CUSUM) (8 features)

**Formula**: $S_k = \max(0, S_{k-1} + (x_k - \mu - K))$

```
Ucav_cusum_max             # Max CUSUM statistic for Ucav
PhaseCav_cusum_max         # Max CUSUM for phase
Uci_cusum_max
PhaseUci_cusum_max
A_Ucr_cusum_max
A_Uamp_cusum_max
vide_cusum_max             # Vacuum drift indicator
courant_pickup_cusum_max   # Current buildup indicator
```

**Physical Interpretation**:
- High CUSUM → Sustained shift from baseline
- Indicates precursor anomaly before fault

---

## 5. Rate-of-Change Features (60 features)

### First Derivative (Velocity)
**Formula**: $\dot{s}(t) = \frac{s(t+1) - s(t)}{\Delta t}$

```
Ucav_deriv1_max            # Peak velocity
Ucav_deriv1_mean           # Average rate of change
PhaseCav_deriv1_max        # Phase drift rate
courant_pickup_deriv1_max  # Current rise rate (field emission)
```

**Physical Significance**:
- $\max|\dot{U}_{cav}|$ → Field collapse rate (quench)
- $\max|\dot{I}_{pickup}|$ → Current rise rate (field emission)

---

### Second Derivative (Acceleration)
**Formula**: $\ddot{s}(t) = \frac{s(t+1) - 2s(t) + s(t-1)}{(\Delta t)^2}$

```
Ucav_deriv2_max            # Peak acceleration
Ucav_deriv2_mean
PhaseCav_deriv2_max        # Phase acceleration (detuning transient)
vide_deriv2_max
```

---

## Feature Selection Priorities

### Tier 1: Physics-Motivated (Always Include)
- `ql`, `decay_tau` → Superconducting state
- `detuning_hz` → Resonance offset
- `control_rms_post` → Loop performance
- `{signal}_cusum_max` → Change points

### Tier 2: High Discriminative Power
- Top features by variance (see Cell 5 in notebook)
- Top discriminative features per fault type (see Cell 8)

### Tier 3: Temporal Precursors
- `Ucav_late_slope` → Quench precursor
- `courant_pickup_{segment}_slope` → Field emission
- `vide_var_ratio_late_early` → Vacuum stability

### Tier 4: Rate-of-Change
- `Ucav_deriv1_max` → Collapse rate
- `PhaseCav_deriv1_max` → Detuning rate

---

## Usage in Models

### Traditional ML (Random Forest, XGBoost)
```python
# Load features
X = data['X_scaled']  # Standardized features
y = data['y_binary']  # Fault vs normal

# Or use PCA
X_pca = data['X_pca']  # 189 components (95% variance)
```

### Deep Learning (LSTM, Transformer)
```python
# Use sequences directly
sequences = data['sequences_full']  # (n_events, n_timesteps, n_signals)
```

### Hybrid Approach
```python
# Combine features + sequences
X_features = data['X_scaled']
X_sequences = data['sequences_pretrigger']  # Pre-trigger only
```

---

## References

**LLRF Physics**:
- Schilcher (1998), "Vector Sum Control", DESY Thesis
- Padamsee (2009), "RF Superconductivity", Wiley

**Feature Engineering**:
- Christ et al. (2018), "tsfresh", Neurocomputing

**Fault Diagnosis**:
- PhysRevAccelBeams.26.012801.pdf - GLR signatures
- 2401.15543v1.pdf - CEBAF LSTM precursors

**Change Point Detection**:
- Basseville & Nikiforov (1993), "Detection of Abrupt Changes"

---

**Quick Start**: See `03_feature_exploration.ipynb` for detailed analysis and visualizations.
