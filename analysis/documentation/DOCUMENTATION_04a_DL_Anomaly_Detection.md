# Step 04a: Deep Learning Anomaly Detection - Comprehensive Documentation

**Date**: 2025-12-29
**Pipeline**: SPIRAL2 LLRF Anomaly Detection
**Methods**: Autoencoder & Variational Autoencoder (VAE)

---

## Executive Summary

This analysis presents unsupervised deep learning anomaly detection for SPIRAL2 LLRF fault data using reconstruction-based neural networks.

**Performance Results**:
- **Autoencoder**: Test AUC = 0.6209, PR AUC = 0.5226
- **VAE**: Test AUC = 0.6138, PR AUC = 0.5104

**Key Findings**:
- Both methods achieve ~62% AUC, significantly better than random (50%) but modest overall
- Reconstruction-based anomaly detection provides baseline for novel fault discovery
- Performance gap compared to supervised methods (93% AUC in Steps 05a, 06a) highlights value of labeled data
- Fast GPU training (<100 seconds) enables rapid model iteration

**Dataset**: 7,427 events processed through 1,243 engineered features

**Recommendation**: Use supervised methods (Steps 05a, 06a) for production; reserve autoencoders for exploratory analysis and novel fault discovery

---

## 1. Scientific Context

### 1.1 SPIRAL2 LLRF System Overview

The SPIRAL2 (Système de Production d'Ions Radioactifs en Ligne de 2ème génération) facility at GANIL operates a superconducting linear accelerator using RF cavities at 88 MHz. The Low-Level RF (LLRF) control system maintains precise amplitude and phase control through continuous monitoring of 8 diagnostic channels sampled at ~17 kHz:

**Primary Signals**:
1. **Ucav** - Cavity voltage amplitude (primary RF power indicator)
2. **PhaseCav** - Cavity phase relative to reference
3. **I, Q** - In-phase and Quadrature components (complex RF representation)

**Control Signals**:
4. **Uci** - Control input voltage
5. **DPCI** - Delta phase control input
6. **PhaseUci** - Phase of control input

**Amplifier Monitoring**:
7. **Modulateur** - High-power amplifier status

### 1.2 Fault Types in Dataset

The dataset contains 7 fault categories representing different failure modes:

1. **Seuil pick-up** - Pickup probe voltage threshold exceeded
2. **Coupure externe rapide** - External fast interlock triggered
3. **Absence autorisation RF** - RF authorization signal missing
4. **Seuil de vide** - Vacuum pressure threshold violated
5. **Claquage ou quench cavité** - Cavity breakdown or superconducting quench
6. **Dépassement seuil de sécurité RF** - RF safety limit exceeded
7. **Régulation signal RF hors tolérance** - RF regulation out of tolerance

### 1.3 Feature Engineering Pipeline

From raw time series (typically 500-2000 samples per event), we extract 1,243 statistical features:

**Time Domain Features** (per channel):
- Statistical moments: mean, std, min, max, median, IQR
- Distribution shape: skewness, kurtosis
- Temporal dynamics: zero-crossing rate, autocorrelation

**Frequency Domain Features**:
- FFT peak frequencies and magnitudes
- Spectral centroid, bandwidth, rolloff
- Spectral entropy (measure of frequency complexity)

**Time-Frequency Features**:
- Wavelet transform coefficients (multi-resolution analysis)
- Short-time Fourier transform (STFT) characteristics

**Cross-Channel Features**:
- Pearson and Spearman correlations between channels
- Phase relationships (especially I-Q, Ucav-PhaseCav)
- Transfer function estimates

**Physical Features**:
- RF power estimates: $P \\propto |I + jQ|^2$
- Detuning indicators: phase drift rates
- Control loop stability metrics

---

## 2. Theoretical Foundations

### 2.1 Autoencoders for Anomaly Detection

#### 2.1.1 Conceptual Framework

**Core Principle**: Autoencoders learn to compress and reconstruct data through a bottleneck layer. When trained exclusively or predominantly on normal data, they develop an efficient internal representation of normal patterns. Anomalous inputs, which deviate significantly from training distribution, cannot be accurately reconstructed, yielding high reconstruction error.

**Inductive Bias**: The dimensionality reduction forces the network to learn only the most salient features of the data. Rare patterns (anomalies) are not captured in this compressed representation.

**Why This Works for LLRF**:
- Normal RF operation exhibits repeatable patterns: stable cavity fields, predictable phase evolution, consistent control responses
- The 1,243-dimensional feature space likely contains significant redundancy that autoencoders can exploit
- Faults introduce irregular signatures (sudden voltage drops, phase jumps, erratic control) that break learned compression patterns

**Architecture Philosophy**:
- **Progressive dimensionality reduction**: 1243 → 512 → 256 → 128 → 64
- **Symmetric decoder**: 64 → 128 → 256 → 512 → 1243
- **Non-linear activations**: ReLU enables learning complex, non-linear compression
- **Regularization**: Dropout (0.2-0.3) and BatchNormalization prevent overfitting

#### 2.1.2 Mathematical Formulation

**Problem Setup**:

Given training dataset $\\mathcal{D} = \\{\\mathbf{x}_1, \\ldots, \\mathbf{x}_n\\}$ with $\\mathbf{x}_i \\in \\mathbb{R}^d$ (where $d=1243$ for LLRF features), learn functions:

1. **Encoder** $f_{\\text{enc}}: \\mathbb{R}^d \\to \\mathbb{R}^k$:
   $$\\mathbf{z} = f_{\\text{enc}}(\\mathbf{x}; \\theta_{\\text{enc}})$$

   where $k \\ll d$ is the latent dimension (encoding dim = 64 in our implementation).

2. **Decoder** $f_{\\text{dec}}: \\mathbb{R}^k \\to \\mathbb{R}^d$:
   $$\\hat{\\mathbf{x}} = f_{\\text{dec}}(\\mathbf{z}; \\theta_{\\text{dec}})$$

**Full Autoencoder**:
$$\\hat{\\mathbf{x}} = (f_{\\text{dec}} \\circ f_{\\text{enc}})(\\mathbf{x}; \\theta) = f_{\\text{AE}}(\\mathbf{x}; \\theta)$$

where $\\theta = \\{\\theta_{\\text{enc}}, \\theta_{\\text{dec}}\\}$ are all network parameters.

**Training Objective** (Reconstruction Loss):
$$\\mathcal{L}(\\theta) = \\frac{1}{n} \\sum_{i=1}^{n} \\|\\mathbf{x}_i - f_{\\text{AE}}(\\mathbf{x}_i; \\theta)\\|_2^2$$

**Optimization** (Stochastic Gradient Descent with Adam):
$$\\theta_{t+1} = \\theta_t - \\eta_t \\cdot m_t / (\\sqrt{v_t} + \\epsilon)$$

where $m_t, v_t$ are first and second moment estimates (Adam adaptive learning rate).

**Anomaly Scoring**:

For test sample $\\mathbf{x}$, compute reconstruction error:
$$s(\\mathbf{x}) = \\|\\mathbf{x} - f_{\\text{AE}}(\\mathbf{x}; \\theta^*)\\|_2^2 = \\sum_{j=1}^{d} (x_j - \\hat{x}_j)^2$$

**Decision Rule**:
$$\\hat{y}(\\mathbf{x}) = \\mathbb{1}[s(\\mathbf{x}) > \\tau]$$

where threshold $\\tau$ can be set as:
- Fixed percentile: $\\tau = Q_{0.95}(\\{s(\\mathbf{x}_i)\\}_{i \\in \\mathcal{D}_{\\text{val}}})$
- Contamination-based: top $\\phi$ fraction of scores
- Statistical threshold: $\\tau = \\mu + 3\\sigma$ where $\\mu, \\sigma$ from validation errors

**Evaluation Metrics**:

1. **ROC AUC** (Receiver Operating Characteristic):
   $$\\text{AUC} = P(s(\\mathbf{x}_{\\text{anom}}) > s(\\mathbf{x}_{\\text{norm}}))$$

   Interpretation: Probability that random anomaly scores higher than random normal sample.

2. **PR AUC** (Precision-Recall):
   $$\\text{PR-AUC} = \\int_0^1 \\text{Precision}(r) \\, dr$$

   More informative than ROC AUC for imbalanced datasets.

#### 2.1.3 Implementation Details for LLRF

**Network Architecture**:

```python
# Encoder
Input(1243)
    → Dense(512, ReLU) + BatchNorm + Dropout(0.3)
    → Dense(256, ReLU) + BatchNorm + Dropout(0.3)
    → Dense(128, ReLU) + BatchNorm + Dropout(0.2)
    → Dense(64, ReLU)  # Latent representation

# Decoder (symmetric)
Dense(64)
    → Dense(128, ReLU) + BatchNorm + Dropout(0.2)
    → Dense(256, ReLU) + BatchNorm + Dropout(0.3)
    → Dense(512, ReLU) + BatchNorm + Dropout(0.3)
    → Dense(1243, Linear)  # Reconstruction
```

**Hyperparameters**:
- Optimizer: Adam (lr=0.001)
- Batch size: 64
- Loss: Mean Squared Error (MSE)
- Early stopping: Patience=15 epochs on validation loss
- Learning rate reduction: Factor=0.5, Patience=5 epochs
- Maximum epochs: 100 (typically stops ~60-80)

**Training Strategy**:
1. 70% train, 15% validation, 15% test split (stratified by label)
2. Train on reconstruction task: minimize $\\|\\mathbf{x} - \\hat{\\mathbf{x}}\\|_2^2$
3. Monitor validation loss for early stopping
4. Restore best weights (lowest validation loss)

#### 2.1.4 References

1. **Hinton, G. E., & Salakhutdinov, R. R. (2006)**. "Reducing the Dimensionality of Data with Neural Networks", *Science*, 313(5786), 504-507.
   DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
   *Seminal paper introducing deep autoencoders with pre-training*

2. **Sakurada, M., & Yairi, T. (2014)**. "Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction", *Proceedings of the MLSDA 2014 2nd Workshop on Machine Learning for Sensory Data Analysis*, pp. 4-11.
   DOI: [10.1145/2689746.2689747](https://doi.org/10.1145/2689746.2689747)
   *Application of autoencoders to anomaly detection*

3. **Chalapathy, R., & Chawla, S. (2019)**. "Deep Learning for Anomaly Detection: A Survey", arXiv:1901.03407.
   URL: [https://arxiv.org/abs/1901.03407](https://arxiv.org/abs/1901.03407)
   *Comprehensive survey covering autoencoder-based methods*

4. **Goodfellow, I., Bengio, Y., & Courville, A. (2016)**. *Deep Learning*, MIT Press. Chapter 14: Autoencoders.
   URL: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
   *Authoritative textbook treatment of autoencoder theory*

---

### 2.2 Variational Autoencoders (VAE)

#### 2.2.1 Conceptual Framework

**VAE Philosophy**: Unlike standard autoencoders that learn deterministic point encodings, VAEs learn a **probability distribution** over the latent space. This probabilistic formulation:

1. **Enables generation**: Can sample from latent space to create new data
2. **Provides uncertainty quantification**: Latent codes have associated variance
3. **Regularizes latent space**: KL divergence term ensures smooth, well-structured latent manifold
4. **Improves generalization**: Stochastic sampling acts as implicit regularization

**Why Probabilistic Encoding Matters**:
- Deterministic autoencoders may create "holes" in latent space (regions that don't decode to valid data)
- VAE forces latent space to be continuous and interpretable
- Uncertainty in encoding reflects ambiguity in input (useful for anomaly detection)

**Application to LLRF Anomaly Detection**:
- Normal LLRF events cluster in high-density regions of learned $p(\\mathbf{x})$
- Anomalies have low probability: $p_{\\theta}(\\mathbf{x}_{\\text{anom}}) \\ll p_{\\theta}(\\mathbf{x}_{\\text{norm}})$
- Combined score (reconstruction + KL) quantifies deviation from learned distribution

#### 2.2.2 Mathematical Formulation

**Generative Model Assumption**:

Assume data $\\mathbf{x}$ generated by latent variable model:
$$p(\\mathbf{x}) = \\int p_{\\theta}(\\mathbf{x}|\\mathbf{z}) p(\\mathbf{z}) \\, d\\mathbf{z}$$

where:
- **Prior**: $p(\\mathbf{z}) = \\mathcal{N}(\\mathbf{z}; \\mathbf{0}, \\mathbf{I})$ (isotropic Gaussian)
- **Likelihood**: $p_{\\theta}(\\mathbf{x}|\\mathbf{z})$ parameterized by decoder neural network

**Intractability**: Computing $p(\\mathbf{x})$ requires marginalizing over $\\mathbf{z}$, which is intractable for complex $p_{\\theta}$.

**Variational Approximation**:

Introduce approximate posterior $q_{\\phi}(\\mathbf{z}|\\mathbf{x})$ (encoder) to approximate true posterior $p_{\\theta}(\\mathbf{z}|\\mathbf{x})$:
$$q_{\\phi}(\\mathbf{z}|\\mathbf{x}) = \\mathcal{N}(\\mathbf{z}; \\mu_{\\phi}(\\mathbf{x}), \\text{diag}(\\sigma^2_{\\phi}(\\mathbf{x})))$$

where encoder outputs mean $\\mu_{\\phi}$ and log-variance $\\log \\sigma^2_{\\phi}$.

**Evidence Lower Bound (ELBO)**:

By Jensen's inequality:
$$\\log p(\\mathbf{x}) \\geq \\mathbb{E}_{q_{\\phi}(\\mathbf{z}|\\mathbf{x})}[\\log p_{\\theta}(\\mathbf{x}|\\mathbf{z})] - D_{\\text{KL}}(q_{\\phi}(\\mathbf{z}|\\mathbf{x}) \\| p(\\mathbf{z}))$$

**Loss Function** (negative ELBO):
$$\\mathcal{L}_{\\text{VAE}}(\\theta, \\phi; \\mathbf{x}) = \\underbrace{-\\mathbb{E}_{q_{\\phi}(\\mathbf{z}|\\mathbf{x})}[\\log p_{\\theta}(\\mathbf{x}|\\mathbf{z})}_{\\text{Reconstruction Loss}} + \\underbrace{D_{\\text{KL}}(q_{\\phi}(\\mathbf{z}|\\mathbf{x}) \\| p(\\mathbf{z}))}_{\\text{KL Regularization}}$$

**Practical Form** (Gaussian likelihood assumption):
$$\\mathcal{L}_{\\text{VAE}} = \\|\\mathbf{x} - \\hat{\\mathbf{x}}\\|_2^2 + D_{\\text{KL}}$$

**KL Divergence** (closed form for Gaussians):
$$D_{\\text{KL}} = -\\frac{1}{2} \\sum_{j=1}^{k} \\left(1 + \\log \\sigma_j^2 - \\mu_j^2 - \\sigma_j^2\\right)$$

**Reparameterization Trick** (enables backpropagation through stochastic node):
$$\\mathbf{z} = \\mu + \\sigma \\odot \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})$$

This separates randomness (ε) from parameters (μ, σ), allowing gradients to flow.

**Anomaly Score**:
$$s_{\\text{VAE}}(\\mathbf{x}) = \\|\\mathbf{x} - \\hat{\\mathbf{x}}\\|_2^2 + \\beta \\cdot D_{\\text{KL}}(q_{\\phi}(\\mathbf{z}|\\mathbf{x}) \\| p(\\mathbf{z}))$$

where $\\beta \\in [0, 1]$ weights KL contribution (we use $\\beta=1$).

**Interpretation of Loss Components**:

1. **Reconstruction Term**:
   - Measures how well decoder $p_{\\theta}(\\mathbf{x}|\\mathbf{z})$ reconstructs input
   - High for inputs that don't fit learned generative model

2. **KL Divergence Term**:
   - Regularizes encoder to produce latent codes close to prior $\\mathcal{N}(0, I)$
   - Prevents "posterior collapse" (encoder ignoring input)
   - Ensures smooth, continuous latent space
   - High KL indicates input requires unusual latent representation (potential anomaly)

#### 2.2.3 Implementation Details for LLRF

**Network Architecture**:

```python
# Encoder
Input(1243)
    → Dense(512, ReLU) + BatchNorm + Dropout(0.3)
    → Dense(256, ReLU) + BatchNorm + Dropout(0.3)
    → Dense(128, ReLU) + BatchNorm
    → Split into:
        - μ: Dense(64, Linear)   # Mean of latent distribution
        - log(σ²): Dense(64, Linear)  # Log-variance

# Sampling Layer (reparameterization)
z = μ + σ ⊙ ε,  ε ~ N(0,I)

# Decoder
Dense(64)
    → Dense(128, ReLU) + BatchNorm + Dropout(0.3)
    → Dense(256, ReLU) + BatchNorm + Dropout(0.3)
    → Dense(512, ReLU) + BatchNorm
    → Dense(1243, Linear)
```

**Training Details**:
- Optimizer: Adam (lr=0.001)
- Loss: Reconstruction MSE + KL divergence
- Batch size: 64
- Early stopping: patience=15 on validation loss
- Latent dimension: 64 (smaller than autoencoder to force stronger regularization)

**Keras 3 Implementation** (custom training loop):

```python
class VAEModel(keras.Model):
    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Reconstruction loss
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=1)
            )

            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )

            total_loss = recon_loss + kl_loss

        # Backpropagation
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
```

#### 2.2.4 References

1. **Kingma, D. P., & Welling, M. (2013)**. "Auto-Encoding Variational Bayes", *International Conference on Learning Representations (ICLR)*.
   URL: [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
   *Original VAE paper introducing variational inference for generative models*

2. **Rezende, D. J., Mohamed, S., & Wierstra, D. (2014)**. "Stochastic Backpropagation and Approximate Inference in Deep Generative Models", *ICML 2014*.
   URL: [https://arxiv.org/abs/1401.4082](https://arxiv.org/abs/1401.4082)
   *Concurrent development of VAE framework with emphasis on inference*

3. **An, J., & Cho, S. (2015)**. "Variational Autoencoder based Anomaly Detection using Reconstruction Probability", *Special Lecture on IE*, SNU Data Mining Center.
   *Application of VAE reconstruction probability for anomaly detection*

4. **Xu, H., Chen, W., Zhao, N., et al. (2018)**. "Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications", *WWW 2018*, pp. 187-196.
   DOI: [10.1145/3178876.3185996](https://doi.org/10.1145/3178876.3185996)
   *Practical VAE anomaly detection for time series with seasonal patterns*

5. **Doersch, C. (2016)**. "Tutorial on Variational Autoencoders", arXiv:1606.05908.
   URL: [https://arxiv.org/abs/1606.05908](https://arxiv.org/abs/1606.05908)
   *Excellent pedagogical introduction to VAE theory*

---

## 3. Results Analysis

### 3.1 Performance Summary

**Test Set Metrics**:

| Model | Test AUC | PR AUC | Train Time (s) | Latent Dim |
|-------|----------|--------|----------------|------------|
| Autoencoder | 0.6209 | 0.5226 | 59.9 | 64 |
| VAE | 0.6138 | 0.5104 | 94.6 | 64 |

**Interpretation**:
- Both methods perform ~12% better than random (AUC=0.5)
- Autoencoder marginally outperforms VAE (0.7% AUC difference)
- Training efficient: <100s on GPU for full model
- Performance modest compared to supervised baselines (93% AUC)

**Why Modest Performance?**:
1. **Feature overlap**: Normal and anomalous events may share similar feature distributions
2. **High-dimensional space**: 1,243 features may contain noise that reconstruction-based methods cannot filter
3. **Label ambiguity**: Some faults may be subtle, appearing "normal" in feature space
4. **Training data composition**: 45.6% anomaly rate violates typical <10% assumption for unsupervised methods

### 3.2 Comparison with Supervised Methods

| Method | Type | Test AUC | Training Time |
|--------|------|----------|---------------|
| **Autoencoder** | Unsupervised | 0.621 | 60s |
| **VAE** | Unsupervised | 0.614 | 95s |
| **LSTM Classifier** (Step 05a) | Supervised | 0.933 | 150s |
| **DNN** (Step 06a) | Supervised | 0.969 | 20s |
| **CNN** (Step 06a) | Supervised | 0.978 | 1254s |

**Key Insight**: Labeled data provides **30-35% performance boost** (AUC: 0.62 → 0.93+). For production deployment, supervised methods strongly preferred.

### 3.3 When to Use Unsupervised Methods

**Appropriate Use Cases**:
1. **Novel fault discovery**: Detect new fault types not present in training labels
2. **Minimal labeling effort**: When acquiring labels is expensive/infeasible
3. **Exploratory analysis**: Validate that faults are indeed "anomalous" in feature space
4. **One-class learning**: Train on normal data only (useful for imbalanced scenarios)
5. **Generative modeling**: VAE can synthesize new fault examples for data augmentation

**SPIRAL2 Recommendation**:
- **Primary detector**: Use supervised LSTM/DNN (Steps 05a, 06a)
- **Secondary/exploratory**: Use Isolation Forest (faster, simpler than DL) or VAE for novel fault detection
- **Ensemble**: Combine supervised + unsupervised predictions for robust detection

---

## 4. Detailed Findings

### 4.1 Training Dynamics

**Autoencoder**:
- Converged after ~60 epochs (early stopping triggered)
- Validation loss plateaued, indicating no overfitting
- Smooth training curve suggests stable optimization
- Learning rate reduction helped fine-tune final performance

**VAE**:
- Converged after ~80 epochs
- KL divergence initially high, then decreased (encoder learning meaningful latent codes)
- Reconstruction loss dominates total loss (KL contributes ~10-20%)
- Longer training time due to stochastic sampling overhead

### 4.2 Reconstruction Error Distributions

**Ideal Pattern** (good separation):
- Normal events: Low reconstruction errors (tight distribution)
- Anomalies: High reconstruction errors (shifted right)

**Observed Pattern** (moderate overlap):
- Significant overlap between normal and anomaly distributions
- Many anomalies have reconstruction errors similar to normal events
- Suggests feature representations alone insufficient for perfect discrimination

**Physical Interpretation**:
- Some fault types (e.g., slow vacuum degradation) may evolve gradually, appearing "normal" in snapshot features
- Other faults (e.g., sudden quench) likely have high reconstruction error
- Feature engineering may benefit from temporal context (→ LSTM in Step 05a)

### 4.3 ROC Curve Analysis

**ROC Curve Shape**:
- Both models show convex curves above random diagonal
- Steeper initial slope indicates ability to catch some anomalies with low false positive rate
- Flattening at high recall indicates diminishing returns (hard-to-detect anomalies)

**Operating Point Selection**:
- High precision regime (FPR < 0.1): Can catch ~20-30% of anomalies with <10% false positives
- Balanced regime (FPR ≈ 0.3): Catch ~50% of anomalies
- High recall regime (FPR > 0.5): Catch ~70% but with many false positives

### 4.4 Precision-Recall Analysis

**PR AUC < ROC AUC** (expected for imbalanced data):
- ROC AUC: 0.62
- PR AUC: 0.52

This indicates performance degrades in high-recall regime (many false positives to catch all anomalies).

**Baseline Comparison**:
- Random classifier PR-AUC = prevalence = 0.456 (dataset has 45.6% anomalies)
- Our models achieve PR-AUC ≈ 0.52, only 0.06 above random
- Indicates challenge of this task for unsupervised methods

---

## 5. Conclusions and Recommendations

### 5.1 Summary of Findings

✓ **Autoencoder and VAE achieve ~62% AUC** on LLRF anomaly detection
✓ **Significantly better than random** (50% AUC) but **modest overall performance**
✓ **Fast training** (<100s on GPU) enables rapid iteration
✓ **Supervised methods outperform** by 30+ percentage points (93% vs 62% AUC)

### 5.2 Practical Recommendations

**For Production Deployment**:
1. ✅ **Primary System**: Use supervised LSTM/DNN/CNN (Steps 05a, 06a) - achieve >93% AUC
2. ⚠️ **Exploratory Tool**: Use VAE for novel fault discovery and generative modeling
3. ⚠️ **Backup Detector**: Consider Isolation Forest (faster, simpler than DL autoencoders)

**For Research/Development**:
1. **Feature Selection**: Reduce 1,243 features to ~100 most discriminative → may improve unsupervised performance
2. **Temporal Models**: Incorporate time series context (see LSTM Autoencoder in Step 05a)
3. **Semi-Supervised**: Train VAE on normal data only, use KL divergence as anomaly score
4. **Ensemble**: Combine reconstruction error from multiple architectures (shallow + deep autoencoders)

### 5.3 Limitations

1. **No temporal context**: Aggregated features lose time series dynamics
2. **High dimensionality**: 1,243 features may contain noise that hinders reconstruction-based detection
3. **Label ambiguity**: Ground truth labels may not perfectly align with feature-space anomalies
4. **Imbalanced data**: 45.6% anomaly rate violates typical unsupervised assumptions (<10%)

### 5.4 Future Work

1. **LSTM Autoencoder**: Process raw time series directly (implemented in Step 05a)
2. **Attention Mechanisms**: Identify which features/time steps contribute to anomaly scores
3. **Adversarial Training**: Train discriminator to distinguish real vs reconstructed samples
4. **Transfer Learning**: Pre-train on similar accelerator facilities (CEBAF, LCLS), fine-tune on SPIRAL2
5. **Physics-Informed Loss**: Incorporate physical constraints (RF power conservation, phase relationships)

---

## 6. References

### Deep Learning Anomaly Detection

1. **Chalapathy, R., & Chawla, S. (2019)**. "Deep Learning for Anomaly Detection: A Survey", arXiv:1901.03407.
   [https://arxiv.org/abs/1901.03407](https://arxiv.org/abs/1901.03407)

2. **Pang, G., Shen, C., Cao, L., & Hengel, A. V. D. (2021)**. "Deep Learning for Anomaly Detection: A Review", *ACM Computing Surveys*, 54(2), 1-38.
   DOI: [10.1145/3439950](https://doi.org/10.1145/3439950)

### Autoencoders

3. **Hinton, G. E., & Salakhutdinov, R. R. (2006)**. "Reducing the Dimensionality of Data with Neural Networks", *Science*, 313(5786), 504-507.
   DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)

4. **Sakurada, M., & Yairi, T. (2014)**. "Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction", *MLSDA 2014*.
   DOI: [10.1145/2689746.2689747](https://doi.org/10.1145/2689746.2689747)

5. **Goodfellow, I., Bengio, Y., & Courville, A. (2016)**. *Deep Learning*, MIT Press.
   [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

### Variational Autoencoders

6. **Kingma, D. P., & Welling, M. (2013)**. "Auto-Encoding Variational Bayes", *ICLR 2014*.
   [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)

7. **Rezende, D. J., Mohamed, S., & Wierstra, D. (2014)**. "Stochastic Backpropagation and Approximate Inference in Deep Generative Models", *ICML 2014*.
   [https://arxiv.org/abs/1401.4082](https://arxiv.org/abs/1401.4082)

8. **An, J., & Cho, S. (2015)**. "Variational Autoencoder based Anomaly Detection using Reconstruction Probability", Special Lecture on IE, SNU Data Mining Center.

9. **Xu, H., et al. (2018)**. "Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications", *WWW 2018*.
   DOI: [10.1145/3178876.3185996](https://doi.org/10.1145/3178876.3185996)

10. **Doersch, C. (2016)**. "Tutorial on Variational Autoencoders", arXiv:1606.05908.
    [https://arxiv.org/abs/1606.05908](https://arxiv.org/abs/1606.05908)

### Accelerator Applications

11. **Edelen, A. L., et al. (2018)**. "Neural Networks for Modeling and Control of Particle Accelerators", *IEEE Transactions on Nuclear Science*, 63(2), 878-897.
    DOI: [10.1109/TNS.2016.2543203](https://doi.org/10.1109/TNS.2016.2543203)

12. **Tennant, C., et al. (2020)**. "Superconducting Radio-Frequency Cavity Fault Classification Using Machine Learning at Jefferson Laboratory", *Physical Review Accelerators and Beams*, 23(11), 114601.
    DOI: [10.1103/PhysRevAccelBeams.23.114601](https://doi.org/10.1103/PhysRevAccelBeams.23.114601)

---

**End of Documentation**
