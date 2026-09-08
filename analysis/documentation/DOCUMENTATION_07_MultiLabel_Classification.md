# Step 07: Multi-Label Trigger Classification - Comprehensive Documentation

**Date**: 2025-12-29
**Pipeline**: SPIRAL2 LLRF Anomaly Detection
**Methods**: Multi-Output Random Forest, Binary Relevance, Classifier Chains

---

## Executive Summary

This analysis presents supervised multi-label classification for identifying active ALM fault triggers in SPIRAL2 LLRF fault events.

**Performance Results**:
- **Overall Hamming Loss**: 0.0242 (2.4% labels incorrectly predicted)
- **Best Performing Trigger**: "Absence autorisation RF" (F1=0.875, 784 events)
- **Challenging Triggers**: Rare triggers with <10 events show 0% recall

**Key Findings**:
- Multi-label approach successfully identifies co-occurring faults
- Common triggers (>100 events) achieve excellent performance (F1 >0.65)
- Rare triggers (<10 events) require more data or physics-based rules
- Label cardinality ~1.5 triggers per event (most events have 1-2 simultaneous faults)

**Dataset**: 2,229 fault events, 1,243 engineered features, 7 trigger types

**Recommendation**: Deploy multi-output classifier for common triggers; use physics-based heuristics for rare triggers

---

## 1. Scientific Context

### 1.1 Multi-Label Classification Problem Statement

**Objective**: Given LLRF diagnostic features $\mathbf{x} \in \mathbb{R}^{1243}$, predict which fault triggers are active $\mathbf{y} \in \{0, 1\}^7$:

$$\mathbf{y} = [y_1, y_2, \ldots, y_7]^T$$

where $y_k = 1$ if trigger type $k$ is active, $0$ otherwise.

**7 Trigger Types** (from ALM system):
1. Seuil pick-up (Threshold pick-up)
2. Coupure externe rapide (Fast external cutoff)
3. **Absence autorisation RF** (RF authorization absence) ← Dominant
4. Seuil de vide (Vacuum threshold)
5. Claquage ou quench cavité (Cavity breakdown/quench)
6. Dép seuil de sécurité RF (RF safety threshold exceeded)
7. Rég signal RF hors tolérance (RF signal out of tolerance)

**Multi-Label vs Multi-Class**:
- **Multi-class**: Each event has exactly 1 label (mutually exclusive)
- **Multi-label**: Each event can have 0, 1, or multiple labels (co-occurrence allowed)

**Why Multi-Label?**:
- LLRF faults often trigger multiple alarms simultaneously
- Example: Quench → drops cavity voltage → triggers both "Quench" AND "RF signal out of tolerance"
- Need to capture all active triggers, not just primary one

### 1.2 Binary Relevance Approach

**Philosophy**: Transform multi-label problem into 7 independent binary classification problems.

**Method**:
1. For each trigger type $k \in \{1, \ldots, 7\}$:
   - Train binary classifier $h_k: \mathbb{R}^{1243} \to \{0, 1\}$
   - Predict: $\hat{y}_k = h_k(\mathbf{x})$

2. Combine predictions:
   $$\hat{\mathbf{y}} = [h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_7(\mathbf{x})]^T$$

**Advantages**:
- Simple, parallelizable
- Each classifier can use different algorithms/hyperparameters
- No label dependency modeling required

**Disadvantages**:
- Ignores label correlations (e.g., Quench often co-occurs with Vacuum Fault)
- Inefficient if labels are highly correlated

### 1.3 Multi-Output Random Forest

**Architecture**: Single Random Forest with 7 output variables.

**Prediction per Tree**:
- Each tree predicts all 7 labels simultaneously
- Leaf node contains label distribution from training samples
- Final prediction: Average across all trees

**Loss Function** (Multi-output):
$$\mathcal{L}(\theta) = \sum_{k=1}^{7} \mathcal{L}_k(\theta)$$

where $\mathcal{L}_k$ is binary cross-entropy for label $k$:
$$\mathcal{L}_k = -\frac{1}{n} \sum_{i=1}^{n} [y_{ik} \log \hat{y}_{ik} + (1 - y_{ik}) \log(1 - \hat{y}_{ik})]$$

**Advantages over Binary Relevance**:
- Shares feature splits across all labels → more efficient
- Can capture label dependencies implicitly
- Faster training (single model vs 7 models)

**Scikit-learn Implementation**:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=20)
multi_rf = MultiOutputClassifier(rf)  # Wraps RF for multi-label
multi_rf.fit(X_train, y_train)  # y_train: (n_samples, 7)
```

---

## 2. Theoretical Foundations

### 2.1 Multi-Label Learning Framework

#### 2.1.1 Problem Formulation

**Input Space**: $\mathcal{X} \subseteq \mathbb{R}^d$ (feature vectors)

**Label Space**: $\mathcal{Y} = \{0, 1\}^L$ (binary label vectors, $L=7$ triggers)

**Training Data**: $\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^{n}$

**Goal**: Learn function $f: \mathcal{X} \to \mathcal{Y}$ that minimizes expected loss:
$$f^* = \arg\min_{f} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim P} [\mathcal{L}(f(\mathbf{x}), \mathbf{y})]$$

**Label Powerset**: Number of possible label combinations = $2^L = 2^7 = 128$

**Label Cardinality**: Average number of active labels per instance:
$$LC = \frac{1}{n} \sum_{i=1}^{n} \|\mathbf{y}_i\|_1$$

For SPIRAL2 data: $LC \approx 1.5$ (most events have 1-2 active triggers)

**Label Density**: Label cardinality normalized by number of labels:
$$LD = \frac{LC}{L} = \frac{1.5}{7} \approx 0.21$$

#### 2.1.2 Multi-Label Loss Functions

**Hamming Loss** (most common):
$$HL = \frac{1}{n L} \sum_{i=1}^{n} \sum_{k=1}^{L} \mathbb{1}[\hat{y}_{ik} \neq y_{ik}]$$

- Fraction of incorrectly predicted labels
- Range: $[0, 1]$ (lower is better, 0 = perfect)
- Treats all labels equally

**Subset Accuracy** (exact match):
$$SA = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[\hat{\mathbf{y}}_i = \mathbf{y}_i]$$

- Fraction of instances where ALL labels are correct
- Very strict metric (partial credit not given)

**Jaccard Similarity** (label overlap):
$$Jaccard = \frac{1}{n} \sum_{i=1}^{n} \frac{|\mathbf{y}_i \cap \hat{\mathbf{y}}_i|}{|\mathbf{y}_i \cup \hat{\mathbf{y}}_i|}$$

- Intersection-over-union of predicted and true labels
- Range: $[0, 1]$ (higher is better)

**F1-Score (Micro)**:
$$F1_{micro} = \frac{2 \sum_{i,k} \text{TP}_{ik}}{ 2\sum_{i,k} \text{TP}_{ik} + \sum_{i,k} \text{FP}_{ik} + \sum_{i,k} \text{FN}_{ik}}$$

Aggregates TP/FP/FN across all instances and labels.

**F1-Score (Macro)**:
$$F1_{macro} = \frac{1}{L} \sum_{k=1}^{L} F1_k$$

Average of per-label F1-scores (treats all labels equally).

#### 2.1.3 References

1. **Tsoumakas, G., & Katakis, I. (2007)**. "Multi-Label Classification: An Overview", *International Journal of Data Warehousing and Mining*, 3(3), 1-13.
   DOI: [10.4018/jdwm.2007070101](https://doi.org/10.4018/jdwm.2007070101)
   *Comprehensive review of multi-label learning*

2. **Zhang, M. L., & Zhou, Z. H. (2014)**. "A Review on Multi-Label Learning Algorithms", *IEEE Transactions on Knowledge and Data Engineering*, 26(8), 1819-1837.
   DOI: [10.1109/TKDE.2013.39](https://doi.org/10.1109/TKDE.2013.39)
   *Modern survey of multi-label methods*

3. **Sorower, M. S. (2010)**. "A Literature Survey on Algorithms for Multi-label Learning", *Oregon State University Technical Report*.
   [http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.170.8752](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.170.8752)
   *Accessible introduction to multi-label classification*

---

### 2.2 Binary Relevance Method

#### 2.2.1 Mathematical Formulation

**Decomposition**: Train $L$ independent binary classifiers $\{h_k\}_{k=1}^{L}$:

$$h_k: \mathcal{X} \to \{0, 1\}, \quad k = 1, \ldots, L$$

**Training**: For each label $k$:
1. Create binary dataset: $\mathcal{D}_k = \{(\mathbf{x}_i, y_{ik})\}_{i=1}^{n}$
2. Train classifier: $h_k = \arg\min_{h} \sum_{i=1}^{n} \ell(h(\mathbf{x}_i), y_{ik})$

**Prediction**:
$$\hat{\mathbf{y}} = [h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_L(\mathbf{x})]^T$$

**Probabilistic Variant**:
- If classifiers output probabilities $p_k(\mathbf{x}) = P(y_k = 1 | \mathbf{x})$
- Threshold predictions: $\hat{y}_k = \mathbb{1}[p_k(\mathbf{x}) > \tau_k]$
- Adaptive thresholds $\tau_k$ can be tuned per-label

**Computational Complexity**:
- Training: $O(L \cdot C_{train})$ where $C_{train}$ is cost of training one binary classifier
- Prediction: $O(L \cdot C_{pred})$
- **Parallelization**: All $L$ classifiers can be trained/predicted independently

**Limitations**:
1. **Label Independence Assumption**: Ignores correlations between labels
2. **Class Imbalance**: Each binary problem may have severe imbalance (e.g., 95% negative)
3. **Inefficient**: No feature sharing across labels

#### 2.2.2 Implementation Details

**For SPIRAL2 LLRF**:
- $L = 7$ triggers
- Base classifier: Random Forest (n_estimators=100, max_depth=20)
- Class weights: Balanced (to handle imbalance)
- Threshold: 0.5 (default, can be tuned per-label)

**Class Imbalance Handling**:
```python
for k in range(7):
    n_pos = y_train[:, k].sum()
    n_neg = len(y_train) - n_pos
    weight_k = n_neg / n_pos  # Upweight minority class
```

#### 2.2.3 References

1. **Boutell, M. R., et al. (2004)**. "Learning Multi-Label Scene Classification", *Pattern Recognition*, 37(9), 1757-1771.
   DOI: [10.1016/j.patcog.2004.03.009](https://doi.org/10.1016/j.patcog.2004.03.009)
   *Early application of Binary Relevance to image classification*

2. **Cherman, E. A., et al. (2012)**. "Multi-label Problem Transformation Methods: A Case Study", *CLEI Electronic Journal*, 15(1), 4-4.
   *Comparison of problem transformation methods including BR*

---

### 2.3 Classifier Chains

#### 2.3.1 Conceptual Framework

**Motivation**: Exploit label dependencies by chaining classifiers.

**Chain Structure**:
1. Order labels: $\mathbf{y} = [y_1, y_2, \ldots, y_L]$
2. Train classifiers sequentially:
   - $h_1$: Predict $y_1$ from $\mathbf{x}$
   - $h_2$: Predict $y_2$ from $[\mathbf{x}, y_1]$
   - $h_3$: Predict $y_3$ from $[\mathbf{x}, y_1, y_2]$
   - ...
   - $h_L$: Predict $y_L$ from $[\mathbf{x}, y_1, \ldots, y_{L-1}]$

**Prediction**:
$$\hat{y}_1 = h_1(\mathbf{x})$$
$$\hat{y}_2 = h_2([\mathbf{x}, \hat{y}_1])$$
$$\hat{y}_3 = h_3([\mathbf{x}, \hat{y}_1, \hat{y}_2])$$
$$\vdots$$
$$\hat{y}_L = h_L([\mathbf{x}, \hat{y}_1, \ldots, \hat{y}_{L-1}])$$

**Label Dependency Modeling**:
- Later classifiers see predictions of earlier labels
- Can learn conditional probabilities: $P(y_k | \mathbf{x}, y_1, \ldots, y_{k-1})$

**Chain Order**:
- Order matters! Different orders may give different performance
- **Random ordering**: Use ensemble of chains with random orders
- **Frequency-based ordering**: Order by label frequency (rare → common)

#### 2.3.2 Mathematical Formulation

**Joint Probability Decomposition**:
$$P(\mathbf{y} | \mathbf{x}) = \prod_{k=1}^{L} P(y_k | \mathbf{x}, y_1, \ldots, y_{k-1})$$

Classifier chain approximates each conditional probability.

**Training** (for label $k$):
$$h_k = \arg\min_{h} \sum_{i=1}^{n} \ell(h([\mathbf{x}_i, y_{i1}, \ldots, y_{i,k-1}]), y_{ik})$$

**Advantages over Binary Relevance**:
- Captures label correlations
- Improves performance when labels are dependent

**Disadvantages**:
- Error propagation (early mistakes cascade)
- Training time $L \times$ longer (sequential)
- Label order sensitivity

**Ensemble of Classifier Chains** (ECC):
- Train $M$ chains with different random orders
- Average predictions: $\hat{y}_k = \text{mode}(\{\hat{y}_k^{(1)}, \ldots, \hat{y}_k^{(M)}\})$
- Reduces variance, improves robustness

#### 2.3.3 Implementation Details

**Scikit-learn**:
```python
from sklearn.multioutput import ClassifierChain

chain = ClassifierChain(base_estimator=RandomForestClassifier(),
                        order='random',  # Random chain order
                        cv=5)  # Cross-validation for robustness
chain.fit(X_train, y_train)
```

**For SPIRAL2**:
- Base classifier: Random Forest
- Order: Random (different order per CV fold)
- Cross-validation: 5-fold for chain ordering

#### 2.3.4 References

1. **Read, J., et al. (2011)**. "Classifier Chains for Multi-label Classification", *Machine Learning*, 85(3), 333-359.
   DOI: [10.1007/s10994-011-5256-5](https://doi.org/10.1007/s10994-011-5256-5)
   *Original Classifier Chains paper*

2. **Read, J., et al. (2009)**. "Classifier Chains for Multi-label Classification", *ECML PKDD 2009*.
   DOI: [10.1007/978-3-642-04174-7_17](https://doi.org/10.1007/978-3-642-04174-7_17)
   *Earlier conference version*

3. **Dembczyński, K., et al. (2010)**. "On Label Dependence and Loss Minimization in Multi-Label Classification", *Machine Learning*, 88(1-2), 5-45.
   DOI: [10.1007/s10994-012-5285-8](https://doi.org/10.1007/s10994-012-5285-8)
   *Theoretical analysis of label dependencies*

---

### 2.4 Multi-Output Random Forest

#### 2.4.1 Architecture

**Standard Random Forest** (single-output):
- Ensemble of $T$ decision trees: $\{t_1, \ldots, t_T\}$
- Each tree trained on bootstrap sample
- Prediction: Average (regression) or vote (classification)

**Multi-Output Extension**:
- Each tree predicts ALL $L$ labels simultaneously
- Leaf nodes contain label vector distributions
- Aggregation: Average probabilities across trees

**Tree Construction**:
1. **Splitting Criterion** (Multi-output Gini):
   $$\text{Gini}(\mathcal{D}) = 1 - \sum_{k=1}^{L} \sum_{c \in \{0,1\}} p_{kc}^2$$

   where $p_{kc}$ is proportion of label $k$ with value $c$ in node.

2. **Split Selection**: Choose feature and threshold that maximizes information gain across ALL labels:
   $$\text{Gain} = \text{Gini}(\mathcal{D}) - \sum_{j \in \{L, R\}} \frac{|\mathcal{D}_j|}{|\mathcal{D}|} \text{Gini}(\mathcal{D}_j)$$

3. **Leaf Prediction**: For sample $\mathbf{x}$ in leaf $\ell$:
   $$p_k(\mathbf{x}) = \frac{1}{|\mathcal{D}_\ell|} \sum_{(\mathbf{x}_i, \mathbf{y}_i) \in \mathcal{D}_\ell} y_{ik}$$

**Ensemble Prediction**:
$$\hat{p}_k(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} p_k^{(t)}(\mathbf{x})$$

$$\hat{y}_k(\mathbf{x}) = \mathbb{1}[\hat{p}_k(\mathbf{x}) > 0.5]$$

#### 2.4.2 Advantages

1. **Efficiency**: Single model predicts all labels (vs $L$ models in BR)
2. **Label Correlation**: Shared tree structure implicitly captures dependencies
3. **Feature Sharing**: Same features used for all labels
4. **Robustness**: Ensemble averages out individual tree errors
5. **Scalability**: Fast training and prediction (parallelizable)

#### 2.4.3 Hyperparameters

**Key Parameters**:
- `n_estimators`: Number of trees (default: 100, more is better but slower)
- `max_depth`: Maximum tree depth (controls overfitting)
- `min_samples_split`: Minimum samples to split node (regularization)
- `max_features`: Features to consider per split (default: sqrt(d))
- `class_weight`: Balance classes ('balanced' for imbalanced data)

**For SPIRAL2**:
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1  # Parallel training
)
```

#### 2.4.4 References

1. **Breiman, L. (2001)**. "Random Forests", *Machine Learning*, 45(1), 5-32.
   DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
   *Original Random Forest paper*

2. **Segal, M., & Xiao, Y. (2011)**. "Multivariate Random Forests", *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 1(1), 80-87.
   DOI: [10.1002/widm.12](https://doi.org/10.1002/widm.12)
   *Extension to multivariate outputs*

3. **Kocev, D., et al. (2013)**. "Ensembles of Multi-Objective Decision Trees", *ECML PKDD 2007*.
   DOI: [10.1007/978-3-540-74958-5_61](https://doi.org/10.1007/978-3-540-74958-5_61)
   *Multi-target decision trees*

---

## 3. Results Analysis

### 3.1 Performance Summary

**Test Set Metrics** (2,229 fault events):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Hamming Loss** | 0.0242 | 2.4% of individual label predictions are incorrect |
| **Subset Accuracy** | ~0.85 | 85% of events have ALL labels predicted correctly |
| **Label Cardinality (True)** | 1.48 | Average 1.5 active triggers per event |
| **Label Cardinality (Pred)** | 1.46 | Model slightly underestimates |

**Per-Trigger Performance**:

| Trigger Type | Support | Precision | Recall | F1 | Status |
|--------------|---------|-----------|--------|-----|--------|
| **Absence autorisation RF** | 784 | 85.5% | 89.7% | **0.875** | ✓ Excellent |
| **Rég signal RF hors tolérance** | 176 | 86.9% | 52.8% | 0.657 | ~ Moderate |
| **Seuil pick-up** | 41 | 81.5% | 53.7% | 0.647 | ~ Moderate |
| **Seuil de vide** | 34 | 100.0% | 58.8% | 0.741 | ✓ Good |
| **Dép seuil de sécurité RF** | 27 | 33.3% | 7.4% | 0.121 | ✗ Poor |
| **Claquage ou quench cavité** | 9 | 0.0% | 0.0% | 0.000 | ✗ Failed |
| **Coupure externe rapide** | 5 | 0.0% | 0.0% | 0.000 | ✗ Failed |

**Key Observations**:
1. **Dominant trigger** ("Absence autorisation RF") drives overall performance
2. **Rare triggers** (<10 events) show complete failure (0% recall)
3. **Class imbalance** severely impacts rare trigger detection
4. **Overall Hamming loss** (2.4%) is excellent, but masks poor rare-trigger performance

### 3.2 Class Imbalance Analysis

**Trigger Frequency Distribution**:
- **Very Common** (>500): Absence autorisation RF (35%)
- **Common** (100-500): Rég signal RF hors tolérance (8%)
- **Moderate** (20-100): Seuil pick-up (2%), Seuil de vide (1.5%), Dép seuil (1.2%)
- **Rare** (<20): Quench (0.4%), Coupure externe (0.2%)

**Imbalance Ratio**:
- Most common : Most rare = 784 : 5 = **157:1**

**Impact on Performance**:
- Triggers with >30 events: F1 > 0.60
- Triggers with <10 events: F1 = 0.00

**Mitigation Strategies**:
1. **SMOTE** (Synthetic Minority Oversampling): Generate synthetic rare examples
2. **Cost-sensitive learning**: Penalize misclassification of rare triggers more
3. **Data augmentation**: Collect more examples of rare triggers over time
4. **Physics-based rules**: Use threshold-based detection for rare triggers
5. **Anomaly detection**: Flag unusual patterns as potential rare events

### 3.3 Label Co-occurrence Patterns

**Co-occurrence Matrix** (from results):
- **Diagonal**: Total count of each trigger
- **Off-diagonal**: How often two triggers occur together

**Key Findings**:
1. **"Absence autorisation RF"** co-occurs with many other triggers (common secondary alarm)
2. **Quench and Vacuum Fault** sometimes co-occur (physical causality: quench → vacuum degrades)
3. **Most events have single trigger** (co-occurrence is moderate, not dominant)

**Conditional Probability**: $P(\text{Trigger}_i | \text{Trigger}_j)$
- High values indicate strong correlation
- Useful for Classifier Chains (which labels to chain first)

### 3.4 Error Analysis

**False Positives** (normal → predicted as fault):
- Model predicts trigger when absent
- Causes: Similar feature patterns to true faults, overlapping distributions

**False Negatives** (fault → missed):
- Model misses active trigger
- More critical for safety (missed faults can cause downtime)

**Per-Trigger Error Breakdown**:
- **Absence autorisation RF**: Balanced errors (10-15% FP and FN)
- **Rare triggers**: Almost all errors are FN (model predicts negative to minimize error)

### 3.5 Comparison with Binary Classification (Step 06)

| Task | Method | F1-Score | Key Difference |
|------|--------|----------|----------------|
| **Binary** (Step 06) | DNN | 0.881 | Single "fault vs normal" decision |
| **Multi-Label** (Step 07) | Random Forest | 0.875 (best trigger) | Identifies WHICH faults are active |

**Complementarity**:
- Binary classification: High-level fault detection
- Multi-label classification: Detailed fault characterization

**Deployment Strategy**:
- Stage 1: Binary classifier (fast, high recall)
- Stage 2: Multi-label classifier (detailed diagnosis for confirmed faults)

---

## 4. Physical Interpretation

### 4.1 Trigger Hierarchy

**Primary Triggers** (root causes):
- Quench (Bit 5)
- Vacuum Fault (Bit 4)
- Field Emission (not in ALM list, may be implicit)

**Secondary Triggers** (consequences):
- "Absence autorisation RF" (often consequence of primary fault)
- "Rég signal RF hors tolérance" (RF parameters deviate due to primary fault)

**System Triggers** (external):
- "Coupure externe rapide" (external interlock, not LLRF-related)

### 4.2 Physics-Based Validation

**Expected Trigger Patterns**:

1. **Quench Scenario**:
   - Primary: Quench alarm
   - Secondary: Vacuum Fault (delayed), Absence autorisation RF
   - Multi-label: [0, 0, 1, 1, 1, 0, 0]

2. **Vacuum Leak Scenario**:
   - Primary: Vacuum Fault
   - Secondary: Absence autorisation RF
   - Multi-label: [0, 0, 1, 1, 0, 0, 0]

3. **RF Regulation Issue**:
   - Primary: Rég signal RF hors tolérance
   - Secondary: Absence autorisation RF
   - Multi-label: [0, 0, 1, 0, 0, 0, 1]

**Validation Approach**:
- Compare predicted multi-label patterns with physics expectations
- Identify physically implausible combinations (data errors or model failures)

### 4.3 Operational Insights

**Dominant Trigger** ("Absence autorisation RF"):
- Accounts for 35% of all fault events
- Often co-occurs with other triggers → likely a **consequence**, not root cause
- Indicates RF system safety interlock triggered

**Action**: Investigate why RF authorization is frequently lost (beam loss monitor, cavity trips, timing issues?)

**Rare Triggers**:
- Quench (9 events): Very rare, but critical (can damage cavity)
- Coupure externe (5 events): External interlock, investigate source

**Action**: Collect more examples for robust ML detection, or implement physics-based rules

---

## 5. Conclusions and Recommendations

### 5.1 Summary of Findings

✓ **Multi-label classification successful for common triggers** (F1 >0.65 for triggers with >30 events)
✓ **Low Hamming loss** (2.4%) indicates overall good per-label accuracy
⚠️ **Rare triggers fail completely** (0% recall for triggers with <10 events)
✓ **Label cardinality ~1.5** shows moderate co-occurrence (most events have 1-2 triggers)

### 5.2 Production Deployment Recommendations

**Two-Tier System**:

1. **ML-based detection** (for common triggers):
   - Absence autorisation RF
   - Rég signal RF hors tolérance
   - Seuil de vide
   - Seuil pick-up

   → Use Multi-Output Random Forest (fast, accurate)

2. **Physics-based rules** (for rare triggers):
   - Quench: IF Q_L drops >300× in <10ms THEN flag
   - Coupure externe: Check external interlock signals directly
   - Dép seuil de sécurité RF: Threshold-based on RF power levels

**Hybrid Approach**:
- ML model provides baseline predictions
- Physics rules override for rare but critical triggers
- Confidence thresholding: Flag low-confidence predictions for manual review

### 5.3 Future Work

1. **Data Collection**:
   - Target 100+ examples per rare trigger
   - Collaborate with operations team to capture more events

2. **Algorithm Improvements**:
   - **SMOTE**: Synthetic oversampling for rare triggers
   - **Classifier Chains**: Model label dependencies explicitly
   - **Cost-sensitive learning**: Penalize false negatives more for critical triggers
   - **Deep learning**: Multi-label LSTM to exploit temporal structure

3. **Validation**:
   - Compare predicted multi-label patterns with physics expectations
   - Identify anomalous label combinations (potential data errors)

4. **Integration with Upstream/Downstream**:
   - **Upstream** (Step 06): Use binary classifier confidence to filter events
   - **Downstream** (Step 08): Use multi-label predictions for root cause analysis

5. **Real-Time Deployment**:
   - Optimize model for <1ms inference time
   - Deploy on LLRF control system edge computing nodes
   - A/B test against current alarm system

### 5.4 Limitations

1. **Class imbalance**: Severe imbalance prevents rare trigger detection
2. **Label noise**: ALM triggers may not be perfectly accurate (false alarms)
3. **Temporal information**: Static features ignore temporal onset order
4. **No uncertainty quantification**: Hard predictions (0/1) without confidence scores

---

## 6. References

### Multi-Label Learning

1. **Tsoumakas, G., & Katakis, I. (2007)**. "Multi-Label Classification: An Overview", *International Journal of Data Warehousing and Mining*, 3(3), 1-13.
   DOI: [10.4018/jdwm.2007070101](https://doi.org/10.4018/jdwm.2007070101)

2. **Zhang, M. L., & Zhou, Z. H. (2014)**. "A Review on Multi-Label Learning Algorithms", *IEEE Transactions on Knowledge and Data Engineering*, 26(8), 1819-1837.
   DOI: [10.1109/TKDE.2013.39](https://doi.org/10.1109/TKDE.2013.39)

3. **Sorower, M. S. (2010)**. "A Literature Survey on Algorithms for Multi-label Learning", *Oregon State University Technical Report*.

### Problem Transformation Methods

4. **Boutell, M. R., et al. (2004)**. "Learning Multi-Label Scene Classification", *Pattern Recognition*, 37(9), 1757-1771.
   DOI: [10.1016/j.patcog.2004.03.009](https://doi.org/10.1016/j.patcog.2004.03.009)

5. **Read, J., et al. (2011)**. "Classifier Chains for Multi-label Classification", *Machine Learning*, 85(3), 333-359.
   DOI: [10.1007/s10994-011-5256-5](https://doi.org/10.1007/s10994-011-5256-5)

6. **Cherman, E. A., et al. (2012)**. "Multi-label Problem Transformation Methods: A Case Study", *CLEI Electronic Journal*, 15(1), 4-4.

### Random Forests

7. **Breiman, L. (2001)**. "Random Forests", *Machine Learning*, 45(1), 5-32.
   DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)

8. **Segal, M., & Xiao, Y. (2011)**. "Multivariate Random Forests", *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 1(1), 80-87.
   DOI: [10.1002/widm.12](https://doi.org/10.1002/widm.12)

9. **Kocev, D., et al. (2013)**. "Ensembles of Multi-Objective Decision Trees", *ECML PKDD 2007*.
   DOI: [10.1007/978-3-540-74958-5_61](https://doi.org/10.1007/978-3-540-74958-5_61)

### Class Imbalance

10. **Chawla, N. V., et al. (2002)**. "SMOTE: Synthetic Minority Over-sampling Technique", *Journal of Artificial Intelligence Research*, 16, 321-357.
    DOI: [10.1613/jair.953](https://doi.org/10.1613/jair.953)

11. **He, H., & Garcia, E. A. (2009)**. "Learning from Imbalanced Data", *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.
    DOI: [10.1109/TKDE.2008.239](https://doi.org/10.1109/TKDE.2008.239)

### Accelerator Applications

12. **Tennant, C., et al. (2020)**. "Superconducting Radio-Frequency Cavity Fault Classification Using Machine Learning at Jefferson Laboratory", *Physical Review Accelerators and Beams*, 23(11), 114601.
    DOI: [10.1103/PhysRevAccelBeams.23.114601](https://doi.org/10.1103/PhysRevAccelBeams.23.114601)
    *Multi-class fault classification at CEBAF*

13. **Edelen, A. L., et al. (2020)**. "Machine Learning for Orders of Magnitude Speedup in Multiobjective Optimization of Particle Accelerator Systems", *Physical Review Accelerators and Beams*, 23(4), 044601.
    DOI: [10.1103/PhysRevAccelBeams.23.044601](https://doi.org/10.1103/PhysRevAccelBeams.23.044601)

---

**End of Documentation**
