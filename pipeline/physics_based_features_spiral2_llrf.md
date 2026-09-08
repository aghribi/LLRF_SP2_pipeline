
# Physics-Based Feature Specification  
## LLRF Post-Mortem Analysis — SPIRAL2 CW LINAC

This document defines a **physics-consistent, CW-compatible set of physics-based features**
for **Low-Level RF (LLRF) post-mortem analysis** of the **SPIRAL2 superconducting linac**.

The goal is to encode **observable cavity–LLRF dynamics**, not to infer intrinsic cavity parameters
that are inaccessible in CW operation.

---

## 0. Scope and Validity

### Applicable Regime
- CW superconducting cavities
- Strongly over-coupled operation
- Closed-loop LLRF control
- Post-mortem data with pre-trigger and transient windows

### Explicit Exclusions
- Direct estimation of intrinsic quality factor $Q_0$
- Interpretation of transient decay as free cavity decay unless:
  - RF drive is OFF
  - LLRF loops are OPEN
  - Beam loading is ZERO

---

## 1. Transient Energy Decay Indicators  
*(Corrected replacement for “$Q_L$” features)*

### 1.1 Physical Motivation

In CW linacs, the cavity voltage decay observed during a fault reflects:
- RF drive removal dynamics
- LLRF loop disengagement
- Protection logic
- Detuning evolution

It **does not** directly reflect intrinsic cavity dissipation.

Nevertheless, **relative decay behaviour** contains strong fault signatures.

---

### 1.2 Phenomenological Model

The cavity envelope is approximated as:

$$
V_{env}(t) \approx V_0 e^{-t / \tau_{eff}}
$$

where:
- $\tau_{eff}$ is an **effective system decay constant**
- It encodes cavity + LLRF + RF chain dynamics

---

### 1.3 Extraction Procedure

1. Identify RF shutdown segment (post-trigger)
2. Compute cavity envelope:
   $$
   V_{env}(t) = \sqrt{I(t)^2 + Q(t)^2}
   $$
3. Select monotonic decay window
4. Fit:
   $$
   \ln V_{env}(t) = \ln V_0 - \frac{t}{\tau_{eff}}
   $$

---

### 1.4 Features

| Feature name | Definition |
|-------------|------------|
| `effective_decay_tau` | Fitted decay constant (s) |
| `decay_rate` | $1 / \tau_{eff}$ |
| `decay_fit_r2` | Goodness of fit |
| `decay_relative_to_nominal` | Ratio to reference decay |
| `decay_non_exponentiality` | RMS log-fit residual |

---

### 1.5 Interpretation

| Pattern | Likely Cause |
|-------|-------------|
| Fast decay | Hard RF trip / interlock |
| Slow decay | Controlled RF ramp-down |
| Poor exponential fit | Active loop or detuning dynamics |

---

## 2. Effective Detuning Features

### 2.1 Physical Motivation

Detuning is the **dominant disturbance** in CW SRF operation and directly impacts:
- RF power margin
- Control stability
- Beam quality

---

### 2.2 Mathematical Definition

Effective detuning from phase drift:

$$
\Delta f_{eff} = \frac{1}{2\pi} \frac{d\phi}{dt}
$$

This represents an **effective detuning**, including mechanical, beam-loading,
and control contributions.

---

### 2.3 Extraction Procedure

1. Unwrap cavity phase
2. Select stationary pre-trigger window
3. Linear regression:
   $$
   \phi(t) = \phi_0 + 2\pi \Delta f_{eff} t
   $$

---

### 2.4 Features

| Feature name | Definition |
|-------------|------------|
| `detuning_mean_hz` | Mean detuning |
| `detuning_std_hz` | RMS detuning |
| `detuning_slope_hz_s` | Drift rate |
| `detuning_peak_to_peak` | Max–min |
| `detuning_jump_at_trigger` | Instantaneous change |

---

## 3. Microphonics Spectral Features

### 3.1 Motivation

Microphonics dominate low-frequency phase noise in CW cavities.

---

### 3.2 Signal Definition

Residual phase:

$$
\phi_{res}(t) = \phi(t) - \langle \phi(t) \rangle
$$

---

### 3.3 Features

| Feature | Definition |
|--------|------------|
| `microphonics_rms` | RMS phase noise |
| `microphonics_dom_freq` | Dominant FFT peak |
| `microphonics_band_power_[f1,f2]` | Spectral energy |
| `microphonics_q_factor` | Resonance sharpness |

---

## 4. LLRF Control Loop Performance

### 4.1 Motivation

Faults manifest primarily as **loss of control authority**.

---

### 4.2 Error Signals

Amplitude and phase errors:

$$
e_A(t) = A_{cav}(t) - A_{ref}(t)
$$

$$
e_\phi(t) = \phi_{cav}(t) - \phi_{ref}(t)
$$

---

### 4.3 Features

| Feature | Definition |
|--------|------------|
| `amp_rms_pre`, `phase_rms_pre` | Steady-state error |
| `amp_rms_post`, `phase_rms_post` | Post-trigger error |
| `control_overshoot_amp` | Max transient |
| `control_settling_time` | Time to ±5% |
| `control_saturation_fraction` | Actuator saturation |

---

## 5. Phase Stability Features

### 5.1 Motivation

Phase stability directly impacts beam longitudinal dynamics.

---

### 5.2 Features

| Feature | Definition |
|--------|------------|
| `phase_jitter_rms` | RMS phase jitter |
| `phase_excursion_pp` | Peak-to-peak |
| `phase_noise_slope` | Low-frequency drift |
| `phase_psd_integral` | Integrated PSD |

---

## 6. RF Power Flow and Balance

### 6.1 Motivation

Quenches and RF faults appear first in RF power balance.

---

### 6.2 Features

| Feature | Definition |
|--------|------------|
| `forward_power_mean` | Mean forward power |
| `reflected_power_mean` | Mean reflected power |
| `power_reflection_ratio` | $P_{ref}/P_{fwd}$ |
| `power_jump_at_trigger` | Discontinuity |
| `power_imbalance_rms` | Stability metric |

---

## 7. Validity and Gating Flags

| Flag | Meaning |
|-----|--------|
| `rf_drive_on` | RF present |
| `llrf_loop_closed` | Feedback active |
| `beam_present` | Beam loading |
| `interlock_type` | Protection source |
| `valid_decay_window` | Decay usable |

---

## 8. Feature Design Principles

- Prefer **relative** over absolute values
- Normalize per cavity
- Encode **temporal dynamics**
- Treat cavity + LLRF as a **single dynamical system**

---

## 9. Summary

This feature set:
- Is **physically correct** for CW SRF operation
- Avoids invalid parameter inference
- Encodes dominant failure modes
- Is suitable for ML-based post-mortem analysis
