# ADVANCED EXPLAINABILITY & INTERPRETABILITY SUPPLEMENT

**Project**: SPIRAL2 LLRF Anomaly Detection
**Focus**: Beyond State-of-the-Art Explainability Techniques
**Version**: 1.0
**Date**: December 23, 2025

---

## Table of Contents

1. [Introduction: The Explainability Challenge](#introduction)
2. [Multi-Level Explainability Framework](#multi-level-framework)
3. [Advanced Global Interpretability Methods](#global-interpretability)
4. [Advanced Local Interpretability Methods](#local-interpretability)
5. [Causal Explainability](#causal-explainability)
6. [Physics-Informed Interpretability](#physics-informed)
7. [Uncertainty-Aware Explanations](#uncertainty-aware)
8. [Interactive & Counterfactual Explanations](#interactive-explanations)
9. [Multimodal & Temporal Explanations](#multimodal-temporal)
10. [Human-AI Collaborative Explainability](#collaborative)
11. [Evaluation of Explanations](#evaluation)
12. [Implementation Recommendations](#implementation)

---

## 1. Introduction: The Explainability Challenge {#introduction}

### 1.1 Why Explainability Matters for LLRF

In critical infrastructure like SPIRAL2, **explainability is not optional**:

1. **Safety**: Operators must understand *why* an alarm was triggered before taking action
2. **Trust**: Domain experts need confidence that ML isn't making spurious correlations
3. **Debugging**: Identify when models fail or learn incorrect patterns
4. **Knowledge Discovery**: Uncover physics insights hidden in data
5. **Regulatory Compliance**: Justify automated decisions in safety-critical systems
6. **Continuous Improvement**: Expert feedback requires understanding model reasoning

### 1.2 Limitations of Standard Methods

Current SOTA (SHAP, LIME, feature importance) has gaps:

- **Post-hoc only**: Explains black boxes without inherent interpretability
- **Feature-centric**: Focuses on statistical features, not physical processes
- **Static explanations**: Doesn't capture temporal dynamics
- **Local vs global trade-off**: Hard to get both simultaneously
- **No causal reasoning**: Correlation ≠ causation
- **Uncertain explanations**: No quantification of explanation reliability
- **Non-interactive**: One-way communication, not dialogue

### 1.3 Our Approach: Multi-Dimensional Explainability

We propose a **comprehensive explainability framework** combining:

```
┌────────────────────────────────────────────────────────────────┐
│                   EXPLAINABILITY DIMENSIONS                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  SCOPE:       Global ←──────────────→ Local                   │
│  TIME:        Post-hoc ←────────────→ Intrinsic               │
│  CAUSALITY:   Correlation ←─────────→ Causal                  │
│  MODALITY:    Features ←────────────→ Physics + Time-Series   │
│  CERTAINTY:   Point Estimate ←──────→ Probabilistic           │
│  INTERACTION: Static ←──────────────→ Interactive              │
│  AUDIENCE:    Technical ←───────────→ Domain Expert           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Multi-Level Explainability Framework {#multi-level-framework}

### 2.1 Hierarchical Explanation Levels

Different stakeholders need different levels of explanation:

#### Level 0: Alert Summary (Operators)
**Question**: *"What happened?"*

```
ALERT: Cavity CMA05-CAV2 - Quench Detected
Time: 2025-12-23 14:32:17
Confidence: 92%
Severity: HIGH
Action Required: Disable RF, inspect cavity
```

#### Level 1: Fault Characteristics (Technicians)
**Question**: *"What are the key indicators?"*

```
Primary Indicators:
- Cavity voltage drop: 8.5 kV → 0.3 kV in 2.1 ms
- Q_L degradation: 4.2×10^5 → 1.8×10^3 (99.6% drop)
- Phase instability: σ_phase = 47°
- No beam loading correlation

Similar Events: 12 quenches in CMA05 last month
```

#### Level 2: Feature Attribution (ML Engineers)
**Question**: *"Which features drove the prediction?"*

```
Top Features (SHAP values):
1. Q_L_drop_rate: +0.43
2. Ucav_decay_time: +0.38
3. Phase_jitter_post: +0.21
4. Ucav_variance_ratio: +0.18
5. Control_error_integral: -0.12 (protective)
```

#### Level 3: Physical Interpretation (Physicists)
**Question**: *"What physical process occurred?"*

```
Interpretation:
The rapid Q_L collapse and exponential voltage decay are consistent
with a hard quench in the cavity's bottom cell. The absence of
beam-induced heating signature suggests field emission as trigger.
Vacuum pressure remained stable, ruling out multipacting.

Likely Sequence:
1. Field emission site activated at high gradient (t=-5ms)
2. Localized heating exceeded critical temperature (t=0ms)
3. Normal-conducting region propagated (t=0-2ms)
4. LLRF detected fault and shut down RF (t=2.1ms)
```

#### Level 4: Causal Chain (Researchers)
**Question**: *"What caused what, and why?"*

```
Causal Graph:
High_Gradient → Field_Emission → Localized_Heating → Quench
     ↑                                    ↓
   Setpoint                         Phase_Instability

Counterfactual: If gradient had been 10% lower, quench probability
would have been <5% (based on causal model).
```

#### Level 5: Actionable Insights (Management)
**Question**: *"What should we do long-term?"*

```
Recommendations:
1. CMA05-CAV2 requires high-pressure water rinsing
2. Reduce operational gradient from 9.5 MV/m to 8.5 MV/m
3. Implement automated gradient reduction protocol when
   Q_L shows early degradation signs
4. Budget for cavity refurbishment in next shutdown

Expected Impact: -40% quench rate, +2% beam availability
```

### 2.2 Explanation Orchestration

**Pipeline**:
```python
def generate_hierarchical_explanation(event, model, level='all'):
    """
    Generate multi-level explanations for a detected anomaly.
    """
    explanations = {}

    # Level 0: Alert
    explanations['alert'] = create_alert_summary(event, model)

    # Level 1: Characteristics
    explanations['characteristics'] = extract_fault_characteristics(event)

    # Level 2: Feature Attribution
    explanations['features'] = compute_shap_values(event, model)

    # Level 3: Physics
    explanations['physics'] = interpret_physics(event, explanations['features'])

    # Level 4: Causal
    explanations['causal'] = causal_analysis(event, causal_graph)

    # Level 5: Recommendations
    explanations['recommendations'] = generate_recommendations(
        explanations['causal'], historical_data
    )

    if level == 'all':
        return explanations
    else:
        return explanations[level]
```

---

## 3. Advanced Global Interpretability Methods {#global-interpretability}

### 3.1 Model Distillation into Rule-Based Systems

**Concept**: Train interpretable surrogate model on complex model's predictions

#### A. Decision Tree Surrogate

```python
from sklearn.tree import DecisionTreeClassifier, export_text

# Train complex model (e.g., XGBoost)
complex_model.fit(X_train, y_train)

# Generate predictions on large dataset
X_large = sample_or_generate_large_dataset(n=100000)
y_complex = complex_model.predict(X_large)

# Train simple decision tree on complex model's behavior
surrogate_tree = DecisionTreeClassifier(
    max_depth=5,  # Force simplicity
    min_samples_leaf=100
)
surrogate_tree.fit(X_large, y_complex)

# Extract human-readable rules
rules = export_text(surrogate_tree, feature_names=feature_names)
print(rules)
```

**Example Output**:
```
|--- Q_L_drop_rate <= 0.85
|   |--- Ucav_variance_ratio <= 1.2
|   |   |--- class: Normal
|   |--- Ucav_variance_ratio > 1.2
|   |   |--- class: RF_Regulation_Issue
|--- Q_L_drop_rate > 0.85
|   |--- Phase_jitter_post <= 15.0
|   |   |--- class: Soft_Quench
|   |--- Phase_jitter_post > 15.0
|   |   |--- class: Hard_Quench
```

**Fidelity Metric**:
```python
fidelity = accuracy_score(y_complex, surrogate_tree.predict(X_large))
print(f"Surrogate fidelity: {fidelity:.3f}")  # Target: >0.90
```

#### B. Rule Extraction via Anchors

```python
from anchor import anchor_tabular

# Initialize explainer
explainer = anchor_tabular.AnchorTabularExplainer(
    class_names=fault_types,
    feature_names=feature_names,
    train_data=X_train.values
)

# Extract rule for specific prediction
explanation = explainer.explain_instance(
    X_test[0],
    complex_model.predict,
    threshold=0.95  # 95% precision
)

print('IF:', ' AND '.join(explanation.names()))
print('THEN: Prediction =', fault_types[explanation.exp_map['prediction']])
print('Precision:', explanation.precision())
print('Coverage:', explanation.coverage())
```

**Example Output**:
```
IF: Q_L_drop_rate > 0.8 AND Ucav_decay_time < 3.2 ms AND vacuum_stable = True
THEN: Prediction = Hard_Quench
Precision: 0.97 (97% of events matching this rule are quenches)
Coverage: 0.23 (23% of all quenches match this rule)
```

### 3.2 Symbolic Regression for Physical Laws

**Concept**: Discover interpretable mathematical relationships

```python
from pysr import PySRRegressor
import sympy

# Target: predict Q_L_drop from signals
X_symbolic = df[['Ucav_initial', 'Ucav_final', 'decay_time', 'gradient']]
y_symbolic = df['Q_L_drop_rate']

# Symbolic regression
model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "sqrt"],
    constraints={
        "exp": 3,  # Max complexity for exp
        "log": 3
    },
    model_selection="best",
)
model.fit(X_symbolic, y_symbolic)

# Get best equation
print(model.sympy())
```

**Example Discovered Equation**:
```
Q_L_drop_rate ≈ exp(-decay_time / 0.0023) * (Ucav_initial / Ucav_final)^0.87

Physical Interpretation:
- Exponential decay matches theoretical quench propagation
- Voltage ratio term indicates energy dissipation
- Time constant ~2.3 ms aligns with thermal diffusion in niobium
```

**Validation**:
```python
y_pred_symbolic = model.predict(X_symbolic)
r2 = r2_score(y_symbolic, y_pred_symbolic)
print(f"Symbolic model R²: {r2:.3f}")

# Compare to black-box
r2_blackbox = r2_score(y_test, xgboost_model.predict(X_test))
print(f"Interpretability gain: {r2:.3f} vs {r2_blackbox:.3f}")
```

### 3.3 Accumulated Local Effects (ALE) Plots

**Advantage over PDP**: Handles correlated features correctly

```python
from alepython import ale_plot

# 1D ALE for single feature
ale_plot(
    X=X_train,
    model=complex_model,
    feature='Q_L_drop_rate',
    bins=50,
    include_CI=True  # Confidence intervals
)
plt.title('Effect of Q_L Drop Rate on Quench Probability')
plt.ylabel('ALE (centered)')

# 2D ALE for feature interactions
ale_plot(
    X=X_train,
    model=complex_model,
    features=['Q_L_drop_rate', 'Phase_jitter_post'],
    bins=20
)
plt.title('Interaction: Q_L Drop × Phase Jitter')
```

**Interpretation**:
- ALE = 0: No effect (after centering)
- ALE > 0: Increases quench probability
- Steep regions: Strong sensitivity
- Flat regions: Insensitive range

### 3.4 Global Sensitivity Analysis

**Sobol Indices**: Decompose variance into feature contributions

```python
from SALib.sample import saltelli
from SALib.analyze import sobol

# Define parameter space
problem = {
    'num_vars': len(feature_names),
    'names': feature_names,
    'bounds': [[X[col].min(), X[col].max()] for col in feature_names]
}

# Generate samples (Saltelli scheme)
param_values = saltelli.sample(problem, N=1024)

# Evaluate model
Y = complex_model.predict_proba(param_values)[:, 1]  # Quench probability

# Compute Sobol indices
Si = sobol.analyze(problem, Y)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# First-order indices (direct effect)
ax1.barh(feature_names, Si['S1'])
ax1.set_xlabel('First-order Sobol Index')
ax1.set_title('Direct Feature Effects')

# Total-order indices (including interactions)
ax2.barh(feature_names, Si['ST'])
ax2.set_xlabel('Total-order Sobol Index')
ax2.set_title('Total Feature Effects (+ Interactions)')
plt.tight_layout()
```

**Insight**: If ST >> S1, feature has strong interactions

---

## 4. Advanced Local Interpretability Methods {#local-interpretability}

### 4.1 Integrated Gradients (IG)

**Concept**: Attribute prediction to input features via path integral

```python
import tensorflow as tf

def integrated_gradients(model, baseline, input_signal, steps=50):
    """
    Compute integrated gradients for time-series input.

    Args:
        model: Keras model
        baseline: Baseline signal (e.g., zeros or mean signal)
        input_signal: Signal to explain
        steps: Number of integration steps
    """
    # Generate interpolated signals
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    interpolated = baseline + alphas[:, tf.newaxis, tf.newaxis] * (input_signal - baseline)

    # Compute gradients at each step
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        predictions = model(interpolated)[:, 1]  # Quench probability

    gradients = tape.gradient(predictions, interpolated)

    # Integrate via trapezoidal rule
    avg_gradients = tf.reduce_mean(gradients, axis=0)
    integrated_grads = (input_signal - baseline) * avg_gradients

    return integrated_grads.numpy()

# Example usage
baseline_signal = np.zeros_like(event_signal)  # or np.mean(X_train, axis=0)
attributions = integrated_gradients(cnn_model, baseline_signal, event_signal)

# Visualize
fig, axes = plt.subplots(8, 1, figsize=(12, 16), sharex=True)
for i, signal_name in enumerate(signal_names):
    axes[i].plot(time, event_signal[:, i], label='Signal', alpha=0.7)
    axes[i].fill_between(time, 0, attributions[:, i], alpha=0.3, label='Attribution')
    axes[i].set_ylabel(signal_name)
    axes[i].legend()
    axes[i].axvline(x=0, color='red', linestyle='--', label='Trigger')
axes[-1].set_xlabel('Time (ms)')
plt.suptitle('Integrated Gradients: Feature Attributions Over Time')
```

**Advantages**:
- Satisfies axioms (sensitivity, implementation invariance)
- Works for any differentiable model
- Time-resolved attributions for sequences

### 4.2 SmoothGrad & Gradient × Input

**Problem**: Gradients can be noisy

**Solution**: Average over noisy samples

```python
def smooth_grad(model, input_signal, noise_level=0.1, n_samples=50):
    """
    Compute smooth gradients by averaging over noisy inputs.
    """
    grads = []
    for _ in range(n_samples):
        noise = np.random.normal(0, noise_level * input_signal.std(), input_signal.shape)
        noisy_input = input_signal + noise

        with tf.GradientTape() as tape:
            tape.watch(noisy_input)
            prediction = model(noisy_input[np.newaxis, ...])[0, 1]

        grad = tape.gradient(prediction, noisy_input)
        grads.append(grad.numpy())

    smooth_grad = np.mean(grads, axis=0)

    # Gradient × Input for scaled attribution
    attribution = smooth_grad * input_signal

    return attribution

attributions_smooth = smooth_grad(cnn_model, event_signal)
```

### 4.3 Layer-wise Relevance Propagation (LRP)

**Concept**: Backpropagate relevance scores through network layers

```python
import innvestigate

# Build LRP analyzer
model_wo_softmax = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output  # Before softmax
)

analyzer = innvestigate.create_analyzer("lrp.epsilon", model_wo_softmax)

# Compute relevance
relevance = analyzer.analyze(event_signal[np.newaxis, ...])

# Visualize
plt.figure(figsize=(12, 8))
plt.imshow(relevance[0].T, aspect='auto', cmap='seismic', vmin=-1, vmax=1)
plt.colorbar(label='Relevance')
plt.xlabel('Time Step')
plt.ylabel('Signal Channel')
plt.yticks(range(8), signal_names)
plt.title('Layer-wise Relevance Propagation')
```

**Interpretation**:
- Red regions: Increase quench prediction
- Blue regions: Decrease quench prediction
- Reveals time-localized contributions

### 4.4 Attention Mechanisms (Built-in Interpretability)

**For Transformer models**: Attention weights are inherently interpretable

```python
# Extract attention from trained Transformer
attention_model = tf.keras.models.Model(
    inputs=transformer.input,
    outputs=transformer.get_layer('multi_head_attention').output[1]  # Attention weights
)

attention_weights = attention_model.predict(event_signal[np.newaxis, ...])

# attention_weights shape: (batch, num_heads, seq_len, seq_len)

# Average over heads
avg_attention = attention_weights.mean(axis=1)[0]  # (seq_len, seq_len)

# Visualize
plt.figure(figsize=(10, 10))
plt.imshow(avg_attention, cmap='viridis')
plt.colorbar(label='Attention Weight')
plt.xlabel('Key Position (Time Step)')
plt.ylabel('Query Position (Time Step)')
plt.title('Self-Attention Map')
plt.axvline(x=3000, color='red', linestyle='--', label='Trigger')
plt.axhline(y=3000, color='red', linestyle='--')
plt.legend()
```

**Insight**: Which time steps attend to which others?

---

## 5. Causal Explainability {#causal-explainability}

### 5.1 Causal Discovery from Data

**Goal**: Learn causal graph structure

#### A. PC Algorithm (Constraint-Based)

```python
from causalnex.structure.notears import from_pandas

# Build causal graph from observational data
sm = from_pandas(
    df[['Q_L', 'Ucav', 'gradient', 'beam_loading', 'quench']],
    tabu_edges=[('quench', 'gradient')],  # Quench cannot cause gradient
    w_threshold=0.3  # Edge weight threshold
)

# Visualize
from causalnex.plots import plot_structure
plot_structure(sm, graph_attributes={'size': '8,8'})
```

#### B. Granger Causality (Time-Series)

```python
from statsmodels.tsa.stattools import grangercausalitytests

# Test if X Granger-causes Y
maxlag = 10
results = grangercausalitytests(
    df[['Ucav', 'Q_L']],  # [Y, X]
    maxlag=maxlag,
    verbose=False
)

# Extract p-values
p_values = [results[i+1][0]['ssr_ftest'][1] for i in range(maxlag)]
print(f"Granger causality (Q_L → Ucav): p={min(p_values):.4f}")
```

**Interpretation**: If p < 0.05, Q_L "Granger-causes" Ucav

### 5.2 Causal Effect Estimation

**Question**: What is the causal effect of increasing gradient on quench probability?

#### A. Structural Causal Model (SCM)

```python
from causalnex.network import BayesianNetwork

# Define structure (from domain knowledge or discovery)
edges = [
    ('gradient', 'field_emission'),
    ('field_emission', 'heating'),
    ('heating', 'quench'),
    ('beam_loading', 'heating'),
    ('Q_L', 'quench')
]
sm = StructureModel(edges)

# Learn conditional probability distributions
bn = BayesianNetwork(sm)
bn = bn.fit_node_states(df)
bn = bn.fit_cpds(df, method="BayesianEstimator", bayes_prior="K2")

# Query: P(quench | do(gradient = high))
from causalnex.inference import InferenceEngine
ie = InferenceEngine(bn)

# Observational (correlation)
p_quench_obs = ie.query()['quench']

# Interventional (causation)
ie_intervened = InferenceEngine(bn)
ie_intervened.do_intervention('gradient', 'high')
p_quench_interv = ie_intervened.query()['quench']

print(f"Observational P(quench | gradient=high): {p_quench_obs:.3f}")
print(f"Interventional P(quench | do(gradient=high)): {p_quench_interv:.3f}")
```

**Insight**: Interventional ≠ Observational if confounders exist

#### B. Propensity Score Matching

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Treatment: high gradient (binary)
df['high_gradient'] = (df['gradient'] > df['gradient'].median()).astype(int)

# Estimate propensity scores (probability of treatment)
ps_model = LogisticRegression()
ps_model.fit(df[['beam_loading', 'Q_L_initial']], df['high_gradient'])
df['propensity'] = ps_model.predict_proba(df[['beam_loading', 'Q_L_initial']])[:, 1]

# Match treated and control units
treated = df[df['high_gradient'] == 1]
control = df[df['high_gradient'] == 0]

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[['propensity']].values)
distances, indices = nn.kneighbors(treated[['propensity']].values)

# Matched control group
matched_control = control.iloc[indices.flatten()]

# Compute Average Treatment Effect (ATE)
ATE = treated['quench'].mean() - matched_control['quench'].mean()
print(f"Average Treatment Effect (high gradient → quench): {ATE:.3f}")
```

### 5.3 Counterfactual Explanations

**Question**: *"What minimal change would have prevented this quench?"*

#### A. Optimization-Based Counterfactuals

```python
from dice_ml import Data, Model, Dice

# Wrap data and model
d = Data(dataframe=df, continuous_features=continuous_cols, outcome_name='quench')
m = Model(model=complex_model, backend='sklearn', model_type='classifier')

# Initialize DiCE
dice_exp = Dice(d, m, method='gradient')

# Generate counterfactual for specific event
query_instance = df[df['event_id'] == 'quench_event_42'].drop(columns=['quench'])

# Find counterfactual: minimal change to avoid quench
cf_examples = dice_exp.generate_counterfactuals(
    query_instance,
    total_CFs=3,  # Number of counterfactuals
    desired_class='opposite',  # Flip prediction
    features_to_vary=['gradient', 'KPI', 'KII']  # Only vary controllable features
)

cf_examples.visualize_as_dataframe()
```

**Example Output**:
```
Original Event:
  gradient: 9.5 MV/m, KPI: 12, quench: YES

Counterfactual 1 (distance: 0.8):
  gradient: 8.3 MV/m (-13%), KPI: 14 (+17%), quench: NO
  Interpretation: Reduce gradient and increase proportional gain

Counterfactual 2 (distance: 1.2):
  gradient: 9.5 MV/m (unchanged), KPI: 25 (+108%), quench: NO
  Interpretation: Aggressive control gain can compensate for high gradient
```

#### B. Physics-Constrained Counterfactuals

```python
def generate_physics_constrained_counterfactual(event, model, constraints):
    """
    Find counterfactual respecting physical constraints.

    Constraints:
    - Power balance: P_forward = P_cavity + P_reflected
    - Phase relationship: reasonable coupling
    - Temporal continuity: smooth signal changes
    """
    from scipy.optimize import minimize

    def objective(x_cf):
        # Minimize distance to original + prediction change
        distance = np.linalg.norm(x_cf - event)
        prediction = model.predict_proba([x_cf])[0, 1]  # Quench probability
        return distance + 10 * prediction  # Penalty for high quench prob

    def constraint_power_balance(x_cf):
        # Extract power-related features
        Ucav, Uci = x_cf[4], x_cf[6]
        P_cav = Ucav**2
        P_in = Uci**2
        # Simplified: P_in ≈ P_cav + losses
        return P_in - P_cav - 0.1 * P_cav  # 10% losses

    def constraint_smooth_transition(x_cf):
        # Ensure signal doesn't jump discontinuously
        # (More complex: would check temporal derivative)
        return 1.0  # Placeholder

    constraints = [
        {'type': 'eq', 'fun': constraint_power_balance},
        {'type': 'ineq', 'fun': constraint_smooth_transition}
    ]

    # Optimize
    result = minimize(
        objective,
        x0=event,
        method='SLSQP',
        constraints=constraints,
        bounds=[(feat.min(), feat.max()) for feat in feature_bounds]
    )

    return result.x

cf_physics = generate_physics_constrained_counterfactual(
    event=quench_event_features,
    model=complex_model,
    constraints=physics_constraints
)
```

**Advantage**: Counterfactuals are physically realizable

---

## 6. Physics-Informed Interpretability {#physics-informed}

### 6.1 Hybrid Physics-ML Models (Inherently Interpretable)

**Architecture**: Embed physical laws in neural network

```python
import tensorflow as tf

class PhysicsInformedNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Neural network branch
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')

        # Output: learned corrections to physics model
        self.correction = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        """
        inputs: [Ucav, Uci, PhaseCav, gradient, ...]
        """
        # Physics-based component (cavity decay model)
        Q_L_physics = self.compute_Q_L_physics(inputs)

        # ML-based correction
        x = self.dense1(inputs)
        x = self.dense2(x)
        correction = self.correction(x)

        # Combine
        Q_L_predicted = Q_L_physics + correction

        return Q_L_predicted, Q_L_physics, correction

    def compute_Q_L_physics(self, inputs):
        """
        Analytical Q_L model based on exponential decay.
        Q_L = ω_0 * τ / 2, where τ = decay time constant
        """
        Ucav = inputs[:, 0]
        time = inputs[:, -1]  # Assume time is last feature

        # Fit exponential: Ucav = U0 * exp(-t/τ)
        # This is simplified; in practice, fit over time window
        tau = -time / tf.math.log(Ucav / Ucav[0] + 1e-10)
        omega_0 = 2 * np.pi * 704.4e6  # Cavity frequency [rad/s]
        Q_L = omega_0 * tau / 2

        return Q_L

# Custom loss: penalize unphysical corrections
def physics_informed_loss(y_true, y_pred_tuple):
    Q_L_pred, Q_L_physics, correction = y_pred_tuple

    # Data fit loss
    mse_loss = tf.reduce_mean((y_true - Q_L_pred)**2)

    # Regularization: prefer small corrections (Occam's razor)
    correction_penalty = 0.1 * tf.reduce_mean(correction**2)

    # Physical constraint: Q_L > 0
    positivity_penalty = tf.reduce_mean(tf.nn.relu(-Q_L_pred))

    return mse_loss + correction_penalty + 10 * positivity_penalty

model = PhysicsInformedNN()
model.compile(optimizer='adam', loss=physics_informed_loss)
```

**Interpretability**:
- **Physics component**: Transparent, based on cavity theory
- **Correction term**: Shows where physics model fails
- **Small corrections**: High confidence in physics
- **Large corrections**: Indicates missing physics or anomaly

**Analysis**:
```python
# Evaluate on test event
Q_L_pred, Q_L_physics, correction = model(test_event)

print(f"Physics model: Q_L = {Q_L_physics:.2e}")
print(f"ML correction: Δ Q_L = {correction:.2e}")
print(f"Final prediction: Q_L = {Q_L_pred:.2e}")
print(f"Ground truth: Q_L = {Q_L_true:.2e}")

if abs(correction / Q_L_physics) > 0.5:
    print("WARNING: Large correction suggests anomaly or model inadequacy")
```

### 6.2 Physics-Guided Feature Engineering

**Dimensionless Numbers**: Physically meaningful ratios

```python
# Example: Define physics-based features
df['energy_ratio'] = (df['Ucav_final']**2) / (df['Ucav_initial']**2)
df['detuning_bandwidth'] = df['detuning'] / df['bandwidth']  # Normalized detuning
df['beam_loading_factor'] = df['beam_current'] * df['Q_ext'] / df['Ucav']
df['thermal_time_ratio'] = df['event_duration'] / 2.3e-3  # Normalized to thermal constant

# Reynolds number analog for RF (if applicable)
df['RF_Reynolds'] = df['gradient'] * df['bandwidth'] / df['losses']
```

**Validation**: Check if models prefer these features

```python
rf_model.fit(X_with_physics_features, y)
importances_physics = rf_model.feature_importances_

# Compare to purely statistical features
rf_model_baseline.fit(X_statistical_only, y)
importances_baseline = rf_model_baseline.feature_importances_

# Interpretation
print("Top physics-based features:")
for feat, imp in sorted(zip(physics_features, importances_physics), key=lambda x: -x[1])[:5]:
    print(f"  {feat}: {imp:.3f}")
```

### 6.3 Residual Analysis for Model Validation

**Concept**: Analyze what the model *doesn't* explain

```python
# Compute residuals
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# Plot residuals vs physics quantities
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].scatter(df_test['gradient'], residuals, alpha=0.3)
axes[0, 0].set_xlabel('Gradient (MV/m)')
axes[0, 0].set_ylabel('Residual')
axes[0, 0].set_title('Residuals vs Gradient')

axes[0, 1].scatter(df_test['Q_L'], residuals, alpha=0.3)
axes[0, 1].set_xlabel('Q_L')
axes[0, 1].set_ylabel('Residual')
axes[0, 1].set_title('Residuals vs Q_L')

# Temporal pattern in residuals?
axes[1, 0].plot(residuals[:100])
axes[1, 0].set_xlabel('Event Index')
axes[1, 0].set_ylabel('Residual')
axes[1, 0].set_title('Residual Time Series')

# Distribution
axes[1, 1].hist(residuals, bins=50, edgecolor='black')
axes[1, 1].set_xlabel('Residual')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Residual Distribution')

plt.tight_layout()
```

**Insights**:
- **Systematic bias**: Residuals correlate with a variable → missing physics
- **Heteroscedasticity**: Residual variance changes → model uncertainty is misestimated
- **Outliers**: Large residuals → potential new anomaly types

---

## 7. Uncertainty-Aware Explanations {#uncertainty-aware}

### 7.1 Conformal Prediction Regions

**Concept**: Provide prediction sets with guaranteed coverage

```python
from nonconformist.cp import IcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc

# Calibration set (separate from train/test)
X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2)

# Train base classifier
base_model = RandomForestClassifier(n_estimators=100)
base_model.fit(X_train, y_train)

# Wrap in conformal predictor
nc = ClassifierNc(base_model, MarginErrFunc())
icp = IcpClassifier(nc)
icp.calibrate(X_cal, y_cal)

# Predict with confidence
significance = 0.1  # 90% confidence
prediction_sets = icp.predict(X_test, significance=significance)

# Explanation with uncertainty
for i in range(5):
    pred_set = [fault_types[j] for j in range(len(fault_types)) if prediction_sets[i, j]]
    print(f"Event {i}: Prediction set (90% confidence): {pred_set}")
```

**Example Output**:
```
Event 0: Prediction set (90% confidence): ['Quench']
  Interpretation: High certainty, single class

Event 1: Prediction set (90% confidence): ['Quench', 'Soft_Quench']
  Interpretation: Uncertain between similar classes

Event 2: Prediction set (90% confidence): ['Quench', 'RF_Regulation', 'Vacuum']
  Interpretation: High uncertainty, multiple plausible explanations
  → Flag for expert review
```

### 7.2 Bayesian Feature Attribution

**Concept**: Quantify uncertainty in SHAP values

```python
# Use multiple bootstrap samples of training data
n_bootstrap = 100
shap_values_bootstrap = []

for b in range(n_bootstrap):
    # Resample training data
    X_boot = X_train.sample(frac=1.0, replace=True)

    # Train model on bootstrap sample
    model_boot = clone(model)
    model_boot.fit(X_boot, y_train.loc[X_boot.index])

    # Compute SHAP for test instance
    explainer = shap.TreeExplainer(model_boot)
    shap_vals = explainer.shap_values(X_test_instance)
    shap_values_bootstrap.append(shap_vals)

# Aggregate: mean and std of SHAP values
shap_mean = np.mean(shap_values_bootstrap, axis=0)
shap_std = np.std(shap_values_bootstrap, axis=0)

# Visualize with error bars
fig, ax = plt.subplots(figsize=(10, 6))
features_sorted = np.argsort(np.abs(shap_mean))[::-1][:20]

ax.barh(
    range(20),
    shap_mean[features_sorted],
    xerr=shap_std[features_sorted],
    capsize=3
)
ax.set_yticks(range(20))
ax.set_yticklabels([feature_names[i] for i in features_sorted])
ax.set_xlabel('SHAP Value ± Std Dev')
ax.set_title('Feature Attribution with Uncertainty')
ax.axvline(x=0, color='black', linestyle='--')
```

**Interpretation**:
- Small error bars: Stable, reliable attribution
- Large error bars: Uncertain, model-dependent
- Features with uncertain attributions: Investigate further

### 7.3 Epistemic vs Aleatoric Uncertainty

**Epistemic** (model uncertainty): Due to limited data/knowledge
**Aleatoric** (data uncertainty): Irreducible randomness

```python
# Using Bayesian Neural Network or Monte Carlo Dropout

# Example: MC Dropout for epistemic uncertainty
def predict_with_uncertainty(model, X, n_iter=100):
    """
    Enable dropout at inference for uncertainty estimation.
    """
    # Assumes model has dropout layers
    predictions = []
    for _ in range(n_iter):
        pred = model(X, training=True)  # Keep dropout active
        predictions.append(pred.numpy())

    predictions = np.array(predictions)

    # Mean prediction
    pred_mean = predictions.mean(axis=0)

    # Epistemic uncertainty (variance across samples)
    epistemic_unc = predictions.var(axis=0)

    # Aleatoric uncertainty (could be modeled as separate output)
    # For simplicity, approximate as prediction entropy
    aleatoric_unc = -np.sum(pred_mean * np.log(pred_mean + 1e-10), axis=1)

    return pred_mean, epistemic_unc, aleatoric_unc

pred, epi_unc, ale_unc = predict_with_uncertainty(nn_model, X_test)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(range(len(pred)), pred[:, 1], c=epi_unc, cmap='Reds')
axes[0].set_xlabel('Event Index')
axes[0].set_ylabel('Quench Probability')
axes[0].set_title('Predictions (color = Epistemic Uncertainty)')
cbar0 = plt.colorbar(axes[0].collections[0], ax=axes[0])
cbar0.set_label('Epistemic Uncertainty')

axes[1].scatter(epi_unc, ale_unc, alpha=0.5)
axes[1].set_xlabel('Epistemic Uncertainty')
axes[1].set_ylabel('Aleatoric Uncertainty')
axes[1].set_title('Uncertainty Decomposition')

# Actionable strategy
high_epi = epi_unc > np.percentile(epi_unc, 90)
high_ale = ale_unc > np.percentile(ale_unc, 90)

print(f"High epistemic uncertainty: {high_epi.sum()} events → Collect more data")
print(f"High aleatoric uncertainty: {high_ale.sum()} events → Inherent ambiguity")
```

---

## 8. Interactive & Counterfactual Explanations {#interactive-explanations}

### 8.1 What-If Tool (Google)

```python
import witwidget
from witwidget.notebook.visualization import WitWidget, WitConfigBuilder

# Prepare data
test_examples = X_test[:100].to_dict(orient='records')

# Configure widget
config_builder = WitConfigBuilder(test_examples)
config_builder.set_custom_predict_fn(lambda x: model.predict_proba(x))
config_builder.set_target_feature('quench')

# Launch interactive tool
WitWidget(config_builder, height=800)
```

**Features**:
- Interactive feature editing (sliders)
- See prediction change in real-time
- Compare multiple datapoints
- Counterfactual search

### 8.2 InterpretML (Microsoft)

**Explainable Boosting Machines (EBM)**: Glass-box model with GAM structure

```python
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

# Train interpretable model
ebm = ExplainableBoostingClassifier(
    interactions=10,  # Allow pairwise interactions
    random_state=42
)
ebm.fit(X_train, y_train)

# Global explanation
ebm_global = ebm.explain_global()
show(ebm_global)

# Local explanation
ebm_local = ebm.explain_local(X_test[:5], y_test[:5])
show(ebm_local)
```

**Visualization**: Automatically generates interactive HTML dashboards

### 8.3 Custom Interactive Dashboard with Plotly Dash

```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("LLRF Anomaly Explanation Explorer"),

    # Event selector
    html.Label("Select Event:"),
    dcc.Dropdown(
        id='event-dropdown',
        options=[{'label': f"Event {i}", 'value': i} for i in range(len(X_test))],
        value=0
    ),

    # Feature sliders (example for one feature)
    html.Label("Adjust Q_L Drop Rate:"),
    dcc.Slider(
        id='q-drop-slider',
        min=0, max=1, step=0.01, value=0.5,
        marks={i/10: f"{i/10}" for i in range(11)}
    ),

    # Prediction display
    html.Div(id='prediction-output'),

    # Signal plot
    dcc.Graph(id='signal-plot'),

    # SHAP explanation
    dcc.Graph(id='shap-plot')
])

@app.callback(
    [Output('prediction-output', 'children'),
     Output('signal-plot', 'figure'),
     Output('shap-plot', 'figure')],
    [Input('event-dropdown', 'value'),
     Input('q-drop-slider', 'value')]
)
def update_dashboard(event_idx, q_drop_modified):
    # Get original event
    event = X_test.iloc[event_idx].copy()

    # Modify feature
    event['Q_L_drop_rate'] = q_drop_modified

    # Predict
    prob = model.predict_proba([event])[0, 1]
    pred_text = f"Quench Probability: {prob:.2%}"

    # Plot signal (placeholder - use real signal data)
    signal_fig = go.Figure()
    signal_fig.add_trace(go.Scatter(x=time, y=signals[event_idx, :, 0], name='Ucav'))
    signal_fig.update_layout(title="Cavity Voltage", xaxis_title="Time (ms)")

    # Compute SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(event)

    shap_fig = go.Figure()
    shap_fig.add_trace(go.Bar(
        x=shap_values[0],
        y=feature_names,
        orientation='h'
    ))
    shap_fig.update_layout(title="Feature Attributions (SHAP)", xaxis_title="SHAP Value")

    return pred_text, signal_fig, shap_fig

if __name__ == '__main__':
    app.run_server(debug=True)
```

---

## 9. Multimodal & Temporal Explanations {#multimodal-temporal}

### 9.1 Time-Series Shapelet Explanations

**Concept**: Find discriminative time-series subsequences

```python
from sktime.classification.shapelet_based import ShapeletTransformClassifier

# Train shapelet-based classifier
stc = ShapeletTransformClassifier(
    n_shapelet_samples=10000,
    max_shapelets=20,
    batch_size=100,
    random_state=42
)
stc.fit(X_train_timeseries, y_train)

# Extract learned shapelets
shapelets = stc.shapelets

# Visualize most discriminative shapelet
best_shapelet = shapelets[0]  # Assume sorted by quality
plt.figure(figsize=(10, 4))
plt.plot(best_shapelet.data)
plt.title(f"Most Discriminative Shapelet (Quality: {best_shapelet.quality:.3f})")
plt.xlabel("Time Step")
plt.ylabel("Amplitude")
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)

# Find shapelet instances in test event
from sktime.distances import dtw_distance

event_signal = X_test_timeseries[0]
distances = []
for i in range(len(event_signal) - len(best_shapelet.data)):
    subsequence = event_signal[i:i+len(best_shapelet.data)]
    dist = dtw_distance(subsequence, best_shapelet.data)
    distances.append(dist)

# Plot event with shapelet match highlighted
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(event_signal, label='Event Signal')
match_idx = np.argmin(distances)
ax1.axvspan(match_idx, match_idx+len(best_shapelet.data), alpha=0.3, color='red', label='Best Match')
ax1.legend()
ax1.set_title('Test Event with Shapelet Match')

ax2.plot(distances)
ax2.set_xlabel('Position')
ax2.set_ylabel('DTW Distance to Shapelet')
ax2.set_title('Shapelet Matching Profile')
```

**Interpretation**: *"This event is classified as a quench because it contains a pattern similar to the canonical quench signature at t=2.3ms"*

### 9.2 Saliency Maps for Time-Series

```python
# For CNN or Transformer on time-series

def compute_saliency_map(model, signal):
    """
    Gradient-based saliency: which time steps are most important?
    """
    signal_tensor = tf.convert_to_tensor(signal[np.newaxis, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(signal_tensor)
        prediction = model(signal_tensor)[0, 1]  # Quench class

    gradient = tape.gradient(prediction, signal_tensor)
    saliency = np.abs(gradient.numpy()[0])

    return saliency

saliency = compute_saliency_map(cnn_model, event_signal)

# Visualize overlay
fig, axes = plt.subplots(8, 1, figsize=(14, 16), sharex=True)
for i, signal_name in enumerate(signal_names):
    # Original signal
    axes[i].plot(time, event_signal[:, i], label='Signal', color='blue', alpha=0.7)

    # Overlay saliency as background color
    axes[i].fill_between(time, 0, saliency[:, i],
                          color='red', alpha=0.3, label='Saliency')
    axes[i].set_ylabel(signal_name)
    axes[i].legend(loc='upper right')
    axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.5)

axes[-1].set_xlabel('Time (ms)')
plt.suptitle('Saliency Map: Important Time Regions for Quench Prediction')
plt.tight_layout()
```

### 9.3 Temporal Attention Explanation

**For Transformer models**:

```python
# Aggregate attention across all heads and layers
# attention_weights: list of (num_heads, seq_len, seq_len) per layer

attention_rollout = np.eye(seq_len)
for layer_attention in attention_weights:
    # Average over heads
    avg_attention = layer_attention.mean(axis=0)

    # Multiply with residual connection
    attention_rollout = avg_attention @ attention_rollout

# attention_rollout[i, j] = importance of time step j for predicting at step i

# Aggregate: importance for final prediction (last time step)
temporal_importance = attention_rollout[-1, :]

# Visualize
plt.figure(figsize=(12, 4))
plt.plot(time, temporal_importance)
plt.xlabel('Time (ms)')
plt.ylabel('Attention Score')
plt.title('Temporal Importance via Attention Rollout')
plt.axvline(x=0, color='red', linestyle='--', label='Trigger')
plt.legend()
```

**Interpretation**: Which historical time steps most influenced the final prediction?

---

## 10. Human-AI Collaborative Explainability {#collaborative}

### 10.1 Active Learning with Explanations

**Concept**: Use explanations to prioritize which samples to label

```python
from modAL.uncertainty import uncertainty_sampling, entropy_sampling
from modAL.models import ActiveLearner

# Initial small labeled set
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=0.9)

# Active learner
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    query_strategy=uncertainty_sampling,
    X_training=X_labeled,
    y_training=y_labeled
)

# Iterative labeling
n_queries = 50
for i in range(n_queries):
    # Query most uncertain sample
    query_idx, query_instance = learner.query(X_unlabeled)

    # Generate explanation for expert
    explainer = shap.TreeExplainer(learner.estimator)
    shap_values = explainer.shap_values(query_instance)

    print(f"\n=== Query {i+1}/{n_queries} ===")
    print(f"Instance: {query_instance}")
    print(f"Top features:")
    for feat, shap_val in sorted(zip(feature_names, shap_values[0]),
                                  key=lambda x: -abs(x[1]))[:5]:
        print(f"  {feat}: {shap_val:.3f}")

    # Get expert label (simulated here)
    true_label = y_unlabeled[query_idx]
    print(f"Expert labels as: {fault_types[true_label]}")

    # Update model
    learner.teach(query_instance, np.array([true_label]))

    # Remove from unlabeled pool
    X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
    y_unlabeled = np.delete(y_unlabeled, query_idx)

print(f"\nFinal model accuracy: {learner.score(X_test, y_test):.3f}")
```

**Benefit**: Explanations help experts understand *why* their label is needed

### 10.2 Explanation-Based Debugging

**Scenario**: Model makes systematic error on specific cavity

```python
# Identify problematic subset
errors = (y_pred != y_test)
problematic_cavity = 'CMA05-CAV2'
problematic_errors = errors & (df_test['ID'] == problematic_cavity)

# Aggregate SHAP values for this subset
shap_problematic = shap_values[problematic_errors]
shap_correct = shap_values[~errors]

# Compare feature importance
import_problematic = np.abs(shap_problematic).mean(axis=0)
import_correct = np.abs(shap_correct).mean(axis=0)

# Rank features by difference
diff = import_problematic - import_correct
suspicious_features = np.argsort(diff)[::-1][:10]

print(f"Features disproportionately important for {problematic_cavity} errors:")
for idx in suspicious_features:
    print(f"  {feature_names[idx]}: Δ importance = {diff[idx]:.3f}")

# Hypothesis: Cavity-specific calibration issue or unique behavior
# → Investigate data quality or add cavity-specific features
```

### 10.3 Contrastive Explanations

**Question**: *"Why was this classified as Quench instead of RF_Regulation?"*

```python
from sklearn.ensemble import RandomForestClassifier
import shap

# Train multi-class model
rf_multi = RandomForestClassifier(n_estimators=100)
rf_multi.fit(X_train, y_train)

# Event classified as Quench
event_idx = 42
event = X_test.iloc[event_idx]
predicted_class = rf_multi.predict([event])[0]
probabilities = rf_multi.predict_proba([event])[0]

print(f"Predicted: {fault_types[predicted_class]}")
print(f"Probabilities: {dict(zip(fault_types, probabilities))}")

# Compute SHAP for quench class
explainer = shap.TreeExplainer(rf_multi)
shap_values_all = explainer.shap_values(event)  # List of arrays, one per class

# Contrastive: Quench vs RF_Regulation
shap_quench = shap_values_all[fault_types.index('Quench')]
shap_rf_reg = shap_values_all[fault_types.index('RF_Regulation')]

# Difference
shap_contrast = shap_quench - shap_rf_reg

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
sorted_idx = np.argsort(np.abs(shap_contrast))[::-1][:15]

ax.barh(range(15), shap_contrast[sorted_idx])
ax.set_yticks(range(15))
ax.set_yticklabels([feature_names[i] for i in sorted_idx])
ax.set_xlabel('SHAP Contrast (Quench - RF_Regulation)')
ax.set_title('Why Quench instead of RF Regulation?')
ax.axvline(x=0, color='black', linestyle='--')

# Annotation
positive_features = [feature_names[i] for i in sorted_idx if shap_contrast[i] > 0][:3]
print(f"Features favoring Quench: {', '.join(positive_features)}")
```

---

## 11. Evaluation of Explanations {#evaluation}

### 11.1 Faithfulness Metrics

**Question**: Do explanations accurately reflect model behavior?

#### A. Perturbation-Based Fidelity

```python
def explanation_fidelity(model, X_test, explanations, top_k=5):
    """
    Remove top-k features identified by explanation.
    Measure drop in model confidence.
    """
    fidelities = []

    for i, (instance, expl) in enumerate(zip(X_test, explanations)):
        # Original prediction
        prob_original = model.predict_proba([instance])[0, 1]

        # Identify top-k features
        top_features = np.argsort(np.abs(expl))[::-1][:top_k]

        # Perturb: set to mean (or zero)
        instance_perturbed = instance.copy()
        instance_perturbed[top_features] = X_train[:, top_features].mean(axis=0)

        # Perturbed prediction
        prob_perturbed = model.predict_proba([instance_perturbed])[0, 1]

        # Fidelity = drop in probability
        fidelity = abs(prob_original - prob_perturbed)
        fidelities.append(fidelity)

    return np.mean(fidelities)

# Compare explanation methods
fidelity_shap = explanation_fidelity(model, X_test, shap_values_test)
fidelity_lime = explanation_fidelity(model, X_test, lime_values_test)

print(f"SHAP fidelity: {fidelity_shap:.3f}")
print(f"LIME fidelity: {fidelity_lime:.3f}")
```

**Interpretation**: Higher fidelity = explanation better captures model's decision

#### B. Infidelity (Continuous Metric)

```python
def infidelity(model, instance, attribution, n_perturbations=100):
    """
    Chun et al. (2020) - Measures explanation-prediction correlation.
    """
    f_x = model.predict_proba([instance])[0, 1]
    infidelities = []

    for _ in range(n_perturbations):
        # Random perturbation
        perturbation = np.random.normal(0, 0.1, size=instance.shape)
        instance_pert = instance + perturbation

        # Model output change
        f_x_pert = model.predict_proba([instance_pert])[0, 1]
        delta_f = f_x_pert - f_x

        # Expected change from explanation
        expected_delta = np.dot(attribution, perturbation)

        # Squared difference
        infidelity_sq = (delta_f - expected_delta)**2
        infidelities.append(infidelity_sq)

    return np.mean(infidelities)

# Lower infidelity = better explanation
```

### 11.2 Stability Metrics

**Question**: Are explanations consistent for similar inputs?

```python
def explanation_stability(model, explainer_fn, instance, n_samples=50, noise_level=0.01):
    """
    Perturb input slightly, measure explanation variance.
    """
    explanations = []

    for _ in range(n_samples):
        noise = np.random.normal(0, noise_level, size=instance.shape)
        instance_noisy = instance + noise

        # Generate explanation
        expl = explainer_fn(model, instance_noisy)
        explanations.append(expl)

    explanations = np.array(explanations)

    # Stability = 1 / (1 + variance)
    variance = explanations.var(axis=0).mean()
    stability = 1 / (1 + variance)

    return stability

# Compare methods
stability_shap = explanation_stability(model, lambda m, x: compute_shap(m, x), X_test[0])
stability_lime = explanation_stability(model, lambda m, x: compute_lime(m, x), X_test[0])

print(f"SHAP stability: {stability_shap:.3f}")
print(f"LIME stability: {stability_lime:.3f}")
```

### 11.3 Human Evaluation

**Gold standard**: Ask domain experts

**Survey Design**:
```
For each explanation, rate (1-5 scale):

1. **Understandability**: Can you understand what the explanation is saying?
2. **Plausibility**: Does the explanation align with your domain knowledge?
3. **Trustworthiness**: Would you trust this explanation to make decisions?
4. **Usefulness**: Does this help you understand the anomaly?
5. **Actionability**: Can you take concrete actions based on this?

Open-ended:
- What additional information would make this explanation more useful?
- Did you discover anything surprising or new from this explanation?
```

**Analysis**:
```python
import pandas as pd

# Survey responses
responses = pd.read_csv('expert_survey.csv')

# Aggregate scores
scores = responses.groupby('explanation_method')[
    ['understandability', 'plausibility', 'trustworthiness', 'usefulness', 'actionability']
].mean()

print(scores)

# Statistical comparison
from scipy.stats import wilcoxon

shap_scores = responses[responses['explanation_method'] == 'SHAP']['usefulness']
lime_scores = responses[responses['explanation_method'] == 'LIME']['usefulness']

stat, p_value = wilcoxon(shap_scores, lime_scores)
print(f"Wilcoxon test (SHAP vs LIME usefulness): p={p_value:.4f}")
```

---

## 12. Implementation Recommendations {#implementation}

### 12.1 Explainability Toolkit

**Recommended Stack**:

```python
# Core explainability libraries
explainability_stack = {
    'global_model_agnostic': ['shap', 'alibi[shap,ale]'],
    'local_model_agnostic': ['lime', 'shap', 'dice-ml'],
    'causal': ['causalnex', 'dowhy', 'econml'],
    'interpretable_models': ['interpret', 'sklearn'],
    'uncertainty': ['nonconformist', 'uncertainty-toolbox'],
    'time_series': ['sktime', 'tslearn', 'stumpy'],
    'physics_informed': ['tensorflow', 'pytorch'],  # Custom implementations
    'visualization': ['plotly', 'matplotlib', 'seaborn']
}
```

### 12.2 Explainability Pipeline Template

```python
class ExplainabilityPipeline:
    """
    Unified pipeline for generating multi-level explanations.
    """

    def __init__(self, model, X_train, feature_names, fault_types):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.fault_types = fault_types

        # Initialize explainers
        self.shap_explainer = shap.TreeExplainer(model)
        self.lime_explainer = lime.LimeTabularExplainer(
            X_train, feature_names=feature_names, class_names=fault_types
        )

    def explain_instance(self, instance, level='all'):
        """
        Generate explanations at multiple levels.
        """
        explanations = {}

        # Feature attribution (Level 2)
        shap_values = self.shap_explainer.shap_values(instance)
        explanations['shap'] = {
            'values': shap_values,
            'top_features': self._get_top_features(shap_values)
        }

        # Physics interpretation (Level 3)
        explanations['physics'] = self._interpret_physics(instance, shap_values)

        # Counterfactual (Level 4)
        explanations['counterfactual'] = self._generate_counterfactual(instance)

        # Uncertainty (cross-cutting)
        explanations['uncertainty'] = self._quantify_uncertainty(instance)

        return explanations

    def _get_top_features(self, shap_values, k=5):
        indices = np.argsort(np.abs(shap_values))[::-1][:k]
        return [(self.feature_names[i], shap_values[i]) for i in indices]

    def _interpret_physics(self, instance, shap_values):
        # Custom logic: map statistical features to physical processes
        physics_interp = {}

        # Example: Q_L drop → quench mechanism
        if 'Q_L_drop_rate' in self.feature_names:
            idx = self.feature_names.index('Q_L_drop_rate')
            if shap_values[idx] > 0.3 and instance[idx] > 0.85:
                physics_interp['mechanism'] = "Hard quench (rapid Q_L collapse)"

        # Add more domain-specific interpretations...

        return physics_interp

    def _generate_counterfactual(self, instance):
        # Use DiCE or custom optimization
        # ... (implementation as shown earlier)
        return {'gradient_reduction': '12%', 'KPI_increase': '+15%'}

    def _quantify_uncertainty(self, instance):
        # Conformal prediction or MC Dropout
        # ... (implementation as shown earlier)
        return {'epistemic': 0.12, 'aleatoric': 0.05}

    def visualize(self, instance, explanations):
        """
        Create comprehensive visualization.
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('SHAP Feature Attribution', 'Signal Time-Series',
                            'Counterfactual Comparison', 'Uncertainty',
                            'Physics Interpretation', 'Attention Map')
        )

        # Add plots to subplots
        # ... (use plotly/matplotlib)

        return fig

# Usage
pipeline = ExplainabilityPipeline(model, X_train, feature_names, fault_types)
explanations = pipeline.explain_instance(X_test[0])
fig = pipeline.visualize(X_test[0], explanations)
fig.show()
```

### 12.3 Integration with Main Pipeline

**In main anomaly detection notebook**:

```python
# After training model and making predictions

from explainability_pipeline import ExplainabilityPipeline

# Initialize
expl_pipeline = ExplainabilityPipeline(
    model=final_model,
    X_train=X_train,
    feature_names=selected_features,
    fault_types=fault_types
)

# For each detected anomaly
for i, event_idx in enumerate(anomaly_indices):
    print(f"\n{'='*60}")
    print(f"Anomaly {i+1}: Event ID {event_idx}")
    print(f"{'='*60}\n")

    instance = X_test[event_idx]

    # Generate explanations
    explanations = expl_pipeline.explain_instance(instance)

    # Display multi-level explanations
    display_explanations(explanations, level='all')

    # Interactive visualization
    fig = expl_pipeline.visualize(instance, explanations)
    fig.show()

    # Optional: request expert feedback
    expert_label = input("Expert, please label this event (or press Enter to skip): ")
    if expert_label:
        # Update training data for next iteration
        update_training_data(instance, expert_label)
```

### 12.4 Documentation & Training

**For operators/engineers**:
- **User guide**: How to interpret dashboards and alerts
- **Case studies**: Examples of different fault types with explanations
- **FAQ**: Common questions about model predictions

**For domain experts**:
- **Technical report**: Methodology, validation, limitations
- **Workshop**: Hands-on session with interactive tools
- **Feedback mechanism**: Channel for reporting issues

**For developers**:
- **API documentation**: Function signatures, parameters
- **Jupyter notebooks**: Reproducible examples
- **Code comments**: Implementation details

---

## Conclusion

This supplement provides a comprehensive toolbox of **advanced explainability techniques** that go beyond standard SHAP/LIME approaches. Key innovations include:

1. **Multi-level explanations** tailored to different audiences
2. **Causal reasoning** to distinguish correlation from causation
3. **Physics-informed interpretability** grounding explanations in domain knowledge
4. **Uncertainty quantification** for explanation reliability
5. **Interactive tools** enabling human-AI collaboration
6. **Temporal & multimodal** explanations for time-series data

### Recommended Implementation Priority:

**Phase 1 (Immediate)**:
- Hierarchical explanations (Levels 0-3)
- SHAP + Integrated Gradients
- Physics-based feature engineering
- Uncertainty via conformal prediction

**Phase 2 (3-6 months)**:
- Causal discovery and SCM
- Counterfactual generation
- Interactive dashboard (Dash/Streamlit)
- Expert validation loop

**Phase 3 (6-12 months)**:
- Physics-informed neural networks
- Symbolic regression
- Active learning with explanations
- Real-time deployment

By systematically implementing these techniques, the SPIRAL2 LLRF anomaly detection system will not only be **accurate** but also **interpretable, trustworthy, and actionable** for operators and domain experts.

---

**References (Additional)**:

- **Causal Inference**: Pearl (2009) "Causality"; Imbens & Rubin (2015)
- **Explainability**: Molnar (2022) "Interpretable Machine Learning"
- **Conformal Prediction**: Shafer & Vovk (2008)
- **Physics-Informed ML**: Karniadakis et al. (2021) Nature Reviews
- **Counterfactuals**: Wachter et al. (2017); Mothilal et al. (2020) DiCE
- **Time-Series Explainability**: Schlegel et al. (2019); Ismail et al. (2020)

---

*Document Status*: Advanced Supplement - Ready for Integration
*Next Steps*: Review with ML team and domain experts, prioritize techniques for Phase 1
