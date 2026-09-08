#!/usr/bin/env python3
"""
Step 04a: Deep Learning Anomaly Detection
-----------------------------------------
Implements Autoencoder and Variational Autoencoder (VAE) for LLRF anomaly detection.
Uses reconstruction error as anomaly score.

This script uses cooked data from features_engineered.pkl and trains:
1. Standard Autoencoder
2. Variational Autoencoder (VAE)

Output: Trained models and reconstruction-based anomaly scores saved to step_04a_dl_anomaly/
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import sys
sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import leakage_safe_split


print("="*80)
print("STEP 04a: DEEP LEARNING ANOMALY DETECTION")
print("="*80)
print(f"\nTensorFlow version: {tf.__version__}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}\n")


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

import os
DATA_DIR = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data'))
OUTPUT_DIR = DATA_DIR / "step_04a_dl_anomaly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Model hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001
PATIENCE = 15  # Early stopping patience


# ============================================================================
# 2. LOAD DATA
# ============================================================================

print("Loading data from features_engineered.pkl...")
data_path = DATA_DIR / "features_engineered.pkl"
with open(data_path, 'rb') as f:
    data = pickle.load(f)

y_binary = data['y_binary']
metadata = data['metadata']

print(f"Data shape: {data['features_all'].shape}")
print(f"Filtered feature columns: {len(data['feature_cols'])}")
print(f"Events: {len(y_binary)}")
print(f"Anomaly rate: {y_binary.mean():.2%}")


# ============================================================================
# 3. PREPARE TRAINING DATA
# ============================================================================

# Leakage-safe train/val/test split: scaler fit on the TRAIN fold only, not
# on the globally pre-fit X_scaled (see leakage_safe_features.py). test_size
# 0.2 of the total + val_size 0.16 of the total reproduces the original
# 64/16/20 train/val/test split (0.2 test, then 0.2 of the remaining 0.8 for
# val).
_split = leakage_safe_split(data, y_binary, test_size=0.2, val_size=0.16,
                             random_state=RANDOM_SEED)
X_train, X_val, X_test = _split['X_train'], _split['X_val'], _split['X_test']
y_train, y_val, y_test = _split['y_train'], _split['y_val'], _split['y_test']

print(f"\nData splits:")
print(f"  Train: {X_train.shape[0]} events ({y_train.mean():.2%} anomalies)")
print(f"  Val:   {X_val.shape[0]} events ({y_val.mean():.2%} anomalies)")
print(f"  Test:  {X_test.shape[0]} events ({y_test.mean():.2%} anomalies)")

input_dim = X_train.shape[1]


# ============================================================================
# 4. AUTOENCODER MODEL
# ============================================================================

class Autoencoder:
    """Standard Autoencoder for anomaly detection"""

    def __init__(self, input_dim, encoding_dim=None):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim or max(16, input_dim // 16)
        self.model = None
        self.history = None

    def build_model(self):
        """Build Autoencoder architecture"""
        # Encoder
        encoder_input = layers.Input(shape=(self.input_dim,), name='encoder_input')
        x = layers.Dense(512, activation='relu')(encoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoding')(x)

        # Decoder
        x = layers.Dense(128, activation='relu')(encoded)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        decoded = layers.Dense(self.input_dim, activation='linear', name='decoder_output')(x)

        # Full autoencoder
        self.model = models.Model(encoder_input, decoded, name='autoencoder')
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )

        return self.model

    def train(self, X_train, X_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the autoencoder"""
        if self.model is None:
            self.build_model()

        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        # Train (reconstruct input from input)
        self.history = self.model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=2
        )

        return self.history

    def predict_reconstruction_error(self, X):
        """Compute reconstruction error as anomaly score"""
        X_reconstructed = self.model.predict(X, verbose=0)
        reconstruction_error = np.mean(np.square(X - X_reconstructed), axis=1)
        return reconstruction_error


# ============================================================================
# 5. VARIATIONAL AUTOENCODER (VAE) MODEL
# ============================================================================

class VAE:
    """Variational Autoencoder for anomaly detection"""

    def __init__(self, input_dim, latent_dim=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim or max(16, input_dim // 16)
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.history = None

    def build_model(self):
        """Build VAE architecture"""
        # Encoder
        encoder_input = layers.Input(shape=(self.input_dim,), name='encoder_input')
        x = layers.Dense(512, activation='relu')(encoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Latent space (mean and log variance)
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)

        # Sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

        # Encoder model
        self.encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name='encoder')

        # Decoder
        decoder_input = layers.Input(shape=(self.latent_dim,), name='decoder_input')
        x = layers.Dense(128, activation='relu')(decoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        decoder_output = layers.Dense(self.input_dim, activation='linear')(x)

        # Decoder model
        self.decoder = models.Model(decoder_input, decoder_output, name='decoder')

        # Full VAE - create a custom model class for Keras 3
        class VAEModel(models.Model):
            def __init__(self, encoder, decoder, **kwargs):
                super().__init__(**kwargs)
                self.encoder = encoder
                self.decoder = decoder
                self.total_loss_tracker = keras.metrics.Mean(name="loss")
                self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
                self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

            def call(self, inputs):
                # Handle both single tensor and (x, y) tuple
                if isinstance(inputs, tuple):
                    inputs = inputs[0]
                z_mean, z_log_var, z = self.encoder(inputs)
                reconstruction = self.decoder(z)
                return reconstruction

            def train_step(self, data):
                # Handle both single tensor and (x, y) tuple
                if isinstance(data, tuple):
                    data = data[0]

                with tf.GradientTape() as tape:
                    z_mean, z_log_var, z = self.encoder(data)
                    reconstruction = self.decoder(z)
                    reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum(tf.square(data - reconstruction), axis=1)
                    )
                    kl_loss = -0.5 * tf.reduce_mean(
                        tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
                    )
                    total_loss = reconstruction_loss + kl_loss

                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                self.total_loss_tracker.update_state(total_loss)
                self.reconstruction_loss_tracker.update_state(reconstruction_loss)
                self.kl_loss_tracker.update_state(kl_loss)
                return {
                    "loss": self.total_loss_tracker.result(),
                    "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                    "kl_loss": self.kl_loss_tracker.result(),
                }

            def test_step(self, data):
                # Handle both single tensor and (x, y) tuple
                if isinstance(data, tuple):
                    data = data[0]

                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.square(data - reconstruction), axis=1)
                )
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
                )
                total_loss = reconstruction_loss + kl_loss

                self.total_loss_tracker.update_state(total_loss)
                self.reconstruction_loss_tracker.update_state(reconstruction_loss)
                self.kl_loss_tracker.update_state(kl_loss)
                return {
                    "loss": self.total_loss_tracker.result(),
                    "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                    "kl_loss": self.kl_loss_tracker.result(),
                }

            @property
            def metrics(self):
                return [
                    self.total_loss_tracker,
                    self.reconstruction_loss_tracker,
                    self.kl_loss_tracker,
                ]

        self.vae = VAEModel(self.encoder, self.decoder, name='vae')
        self.vae.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

        return self.vae

    def train(self, X_train, X_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the VAE"""
        if self.vae is None:
            self.build_model()

        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        # Train (don't pass y when using custom train_step in Keras 3)
        self.history = self.vae.fit(
            X_train,
            validation_data=X_val,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=2
        )

        return self.history

    def predict_reconstruction_error(self, X):
        """Compute reconstruction error as anomaly score"""
        X_reconstructed = self.vae.predict(X, verbose=0)
        reconstruction_error = np.mean(np.square(X - X_reconstructed), axis=1)
        return reconstruction_error


# ============================================================================
# 6. TRAIN MODELS
# ============================================================================

print("\n" + "="*80)
print("TRAINING AUTOENCODER")
print("="*80)

autoencoder = Autoencoder(input_dim=input_dim)
autoencoder.build_model()
print(f"\nAutoencoder architecture:")
autoencoder.model.summary()

print(f"\nTraining Autoencoder...")
start_time = datetime.now()
ae_history = autoencoder.train(X_train, X_val)
ae_train_time = (datetime.now() - start_time).total_seconds()
print(f"Training completed in {ae_train_time:.1f} seconds")


print("\n" + "="*80)
print("TRAINING VARIATIONAL AUTOENCODER (VAE)")
print("="*80)

vae = VAE(input_dim=input_dim)
vae.build_model()
print(f"\nVAE architecture:")
vae.vae.summary()

print(f"\nTraining VAE...")
start_time = datetime.now()
vae_history = vae.train(X_train, X_val)
vae_train_time = (datetime.now() - start_time).total_seconds()
print(f"Training completed in {vae_train_time:.1f} seconds")


# ============================================================================
# 7. EVALUATE MODELS
# ============================================================================

print("\n" + "="*80)
print("EVALUATING MODELS")
print("="*80)

# Compute reconstruction errors
print("\nComputing reconstruction errors...")
ae_train_errors = autoencoder.predict_reconstruction_error(X_train)
ae_val_errors = autoencoder.predict_reconstruction_error(X_val)
ae_test_errors = autoencoder.predict_reconstruction_error(X_test)

vae_train_errors = vae.predict_reconstruction_error(X_train)
vae_val_errors = vae.predict_reconstruction_error(X_val)
vae_test_errors = vae.predict_reconstruction_error(X_test)

# Compute ROC AUC scores (higher reconstruction error = anomaly)
ae_train_auc = roc_auc_score(y_train, ae_train_errors)
ae_val_auc = roc_auc_score(y_val, ae_val_errors)
ae_test_auc = roc_auc_score(y_test, ae_test_errors)

vae_train_auc = roc_auc_score(y_train, vae_train_errors)
vae_val_auc = roc_auc_score(y_val, vae_val_errors)
vae_test_auc = roc_auc_score(y_test, vae_test_errors)

print(f"\nAutoencoder ROC AUC:")
print(f"  Train: {ae_train_auc:.4f}")
print(f"  Val:   {ae_val_auc:.4f}")
print(f"  Test:  {ae_test_auc:.4f}")

print(f"\nVAE ROC AUC:")
print(f"  Train: {vae_train_auc:.4f}")
print(f"  Val:   {vae_val_auc:.4f}")
print(f"  Test:  {vae_test_auc:.4f}")

# Compute precision-recall AUC
ae_precision, ae_recall, _ = precision_recall_curve(y_test, ae_test_errors)
ae_pr_auc = auc(ae_recall, ae_precision)

vae_precision, vae_recall, _ = precision_recall_curve(y_test, vae_test_errors)
vae_pr_auc = auc(vae_recall, vae_precision)

print(f"\nPrecision-Recall AUC (Test):")
print(f"  Autoencoder: {ae_pr_auc:.4f}")
print(f"  VAE:         {vae_pr_auc:.4f}")


# ============================================================================
# 8. SAVE MODELS AND RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODELS AND RESULTS")
print("="*80)

# Save Keras models
ae_model_path = OUTPUT_DIR / "autoencoder_model.keras"
vae_weights_path = OUTPUT_DIR / "vae_weights.weights.h5"

autoencoder.model.save(ae_model_path)
vae.vae.save_weights(vae_weights_path)  # Save weights instead of full model for custom model
print(f"Saved Autoencoder to: {ae_model_path}")
print(f"Saved VAE weights to: {vae_weights_path}")

# Save results
results = {
    'autoencoder': {
        'model_path': str(ae_model_path),
        'train_time': ae_train_time,
        'encoding_dim': autoencoder.encoding_dim,
        'train_errors': ae_train_errors,
        'val_errors': ae_val_errors,
        'test_errors': ae_test_errors,
        'train_auc': ae_train_auc,
        'val_auc': ae_val_auc,
        'test_auc': ae_test_auc,
        'pr_auc': ae_pr_auc,
        'history': ae_history.history
    },
    'vae': {
        'weights_path': str(vae_weights_path),
        'train_time': vae_train_time,
        'latent_dim': vae.latent_dim,
        'train_errors': vae_train_errors,
        'val_errors': vae_val_errors,
        'test_errors': vae_test_errors,
        'train_auc': vae_train_auc,
        'val_auc': vae_val_auc,
        'test_auc': vae_test_auc,
        'pr_auc': vae_pr_auc,
        'history': vae_history.history
    },
    'data_splits': {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'train_anomaly_rate': float(y_train.mean()),
        'val_anomaly_rate': float(y_val.mean()),
        'test_anomaly_rate': float(y_test.mean()),
        'idx_train': _split['idx_train'],
        'idx_val': _split['idx_val'],
        'idx_test': _split['idx_test'],
        'scaler': _split['scaler'],
        'pca': _split['pca'],
        'feature_cols': _split['feature_cols'],
    },
    'labels': {
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
}

results_path = OUTPUT_DIR / "dl_anomaly_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"Saved results to: {results_path}")


# ============================================================================
# 9. GENERATE PLOTS
# ============================================================================

print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

# Plot 1: Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Autoencoder loss
axes[0].plot(ae_history.history['loss'], label='Train Loss', alpha=0.8)
axes[0].plot(ae_history.history['val_loss'], label='Val Loss', alpha=0.8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Autoencoder Training History')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# VAE loss
axes[1].plot(vae_history.history['loss'], label='Train Loss', alpha=0.8)
axes[1].plot(vae_history.history['val_loss'], label='Val Loss', alpha=0.8)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('VAE Training History')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_history.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: training_history.png")

# Plot 2: Reconstruction error distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Autoencoder - Train set
axes[0, 0].hist(ae_train_errors[y_train == 0], bins=50, alpha=0.6, label='Normal', density=True)
axes[0, 0].hist(ae_train_errors[y_train == 1], bins=50, alpha=0.6, label='Anomaly', density=True)
axes[0, 0].set_xlabel('Reconstruction Error')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title(f'Autoencoder - Train Set (AUC={ae_train_auc:.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Autoencoder - Test set
axes[0, 1].hist(ae_test_errors[y_test == 0], bins=50, alpha=0.6, label='Normal', density=True)
axes[0, 1].hist(ae_test_errors[y_test == 1], bins=50, alpha=0.6, label='Anomaly', density=True)
axes[0, 1].set_xlabel('Reconstruction Error')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title(f'Autoencoder - Test Set (AUC={ae_test_auc:.4f})')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# VAE - Train set
axes[1, 0].hist(vae_train_errors[y_train == 0], bins=50, alpha=0.6, label='Normal', density=True)
axes[1, 0].hist(vae_train_errors[y_train == 1], bins=50, alpha=0.6, label='Anomaly', density=True)
axes[1, 0].set_xlabel('Reconstruction Error')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title(f'VAE - Train Set (AUC={vae_train_auc:.4f})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# VAE - Test set
axes[1, 1].hist(vae_test_errors[y_test == 0], bins=50, alpha=0.6, label='Normal', density=True)
axes[1, 1].hist(vae_test_errors[y_test == 1], bins=50, alpha=0.6, label='Anomaly', density=True)
axes[1, 1].set_xlabel('Reconstruction Error')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title(f'VAE - Test Set (AUC={vae_test_auc:.4f})')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "reconstruction_error_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reconstruction_error_distributions.png")


print("\n" + "="*80)
print("STEP 04a COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nKey files generated:")
print(f"  - autoencoder_model.keras")
print(f"  - vae_model.keras")
print(f"  - dl_anomaly_results.pkl")
print(f"  - training_history.png")
print(f"  - reconstruction_error_distributions.png")
print(f"\nBest model: {'Autoencoder' if ae_test_auc > vae_test_auc else 'VAE'}")
print(f"Best Test AUC: {max(ae_test_auc, vae_test_auc):.4f}")
