#!/usr/bin/env python3
"""
Step 05a: Event Sequence LSTM for Temporal Anomaly Detection
------------------------------------------------------------
Implements LSTM models for LLRF temporal anomaly detection.

This script uses cooked data from features_engineered.pkl and trains:
1. LSTM Autoencoder (reconstruction-based anomaly detection)
2. LSTM Binary Classifier (supervised anomaly detection)

Output: Trained models and temporal anomaly scores saved to step_05a_event_sequence_lstm/
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
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    classification_report, accuracy_score, f1_score
)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import raw_feature_matrix


print("="*80)
print("STEP 05a: EVENT SEQUENCE LSTM")
print("="*80)
print(f"\nTensorFlow version: {tf.__version__}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}\n")


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

import os
DATA_DIR = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data'))
OUTPUT_DIR = DATA_DIR / "step_05a_event_sequence_lstm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Model hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001
PATIENCE = 15  # Early stopping patience

# Sequence parameters
SEQUENCE_LENGTH = 10  # Number of timesteps to look back
STRIDE = 1  # How much to shift the window


# ============================================================================
# 2. LOAD DATA
# ============================================================================

print("Loading data from features_engineered.pkl...")
data_path = DATA_DIR / "features_engineered.pkl"
with open(data_path, 'rb') as f:
    data = pickle.load(f)

y_binary = data['y_binary']
metadata = data['metadata']
X_raw, feature_cols = raw_feature_matrix(data)

print(f"Data shape: {X_raw.shape}")
print(f"Events: {X_raw.shape[0]}")
print(f"Anomaly rate: {y_binary.mean():.2%}")


# ============================================================================
# 3. SPLIT EVENTS (BLOCK-WISE), THEN CREATE SEQUENCES WITHIN EACH SPLIT
# ============================================================================
# Original code built sliding-window sequences (each a run of SEQUENCE_LENGTH
# consecutive EVENTS, not signal samples) over the FULL dataset, THEN randomly
# split sequences into train/val/test. Two leakage sources in that order:
#  1. Scaler was fit on the full X_scaled before any split existed.
#  2. Overlapping windows share raw events, so a single event could appear in
#     sequences landing in both train and test after a random sequence-level
#     split -- a second, independent leak from the scaler issue.
# Fixed by reordering: split raw events into CONTIGUOUS BLOCKS first (whole
# blocks randomly assigned to train/val/test, ~64/16/20 proportions), fit the
# scaler on train-block events only, then build sequences separately within
# each split -- since blocks are contiguous and wholly owned by one split, no
# window can straddle a split boundary, and per-event random assignment
# (which would make windows of BLOCK_SIZE consecutive events almost never
# land in one split) is avoided.

from sklearn.preprocessing import StandardScaler

BLOCK_SIZE = 20  # events per block; > SEQUENCE_LENGTH so most blocks yield sequences

n_events = len(y_binary)
n_blocks = int(np.ceil(n_events / BLOCK_SIZE))
block_ids = np.arange(n_blocks)
rng = np.random.RandomState(RANDOM_SEED)
rng.shuffle(block_ids)

n_train_blocks = int(round(0.64 * n_blocks))
n_val_blocks = int(round(0.16 * n_blocks))
train_blocks = set(block_ids[:n_train_blocks].tolist())
val_blocks = set(block_ids[n_train_blocks:n_train_blocks + n_val_blocks].tolist())
test_blocks = set(block_ids[n_train_blocks + n_val_blocks:].tolist())

event_block = np.arange(n_events) // BLOCK_SIZE
idx_train = np.where(np.isin(event_block, list(train_blocks)))[0]
idx_val = np.where(np.isin(event_block, list(val_blocks)))[0]
idx_test = np.where(np.isin(event_block, list(test_blocks)))[0]

scaler = StandardScaler()
X_train_events = scaler.fit_transform(X_raw[idx_train])
X_val_events = scaler.transform(X_raw[idx_val])
X_test_events = scaler.transform(X_raw[idx_test])


def create_sequences_within_blocks(split_idx, X_split, y_full, block_size=BLOCK_SIZE,
                                    sequence_length=SEQUENCE_LENGTH, stride=STRIDE):
    """Build sliding-window sequences from events in ONE split, sliding only
    within each contiguous block (never across a block boundary, so never
    across a split boundary either since blocks are wholly owned by a split)."""
    X_seq, y_seq = [], []
    order = np.argsort(split_idx)
    ordered_positions = split_idx[order]
    X_ordered = X_split[order]

    # Process block by block (positions for one block are contiguous within
    # ordered_positions since split_idx was built from whole blocks).
    unique_blocks = np.unique(ordered_positions // block_size)
    for b in unique_blocks:
        in_block = np.where(ordered_positions // block_size == b)[0]
        block_positions = ordered_positions[in_block]
        X_block = X_ordered[in_block]
        for i in range(0, len(block_positions) - sequence_length + 1, stride):
            X_seq.append(X_block[i:i + sequence_length])
            y_seq.append(y_full[block_positions[i + sequence_length - 1]])

    return np.array(X_seq), np.array(y_seq)


X_train, y_train = create_sequences_within_blocks(idx_train, X_train_events, y_binary)
X_val, y_val = create_sequences_within_blocks(idx_val, X_val_events, y_binary)
X_test, y_test = create_sequences_within_blocks(idx_test, X_test_events, y_binary)

print(f"\nData splits:")
print(f"  Train: {X_train.shape[0]} sequences ({y_train.mean():.2%} anomalies)")
print(f"  Val:   {X_val.shape[0]} sequences ({y_val.mean():.2%} anomalies)")
print(f"  Test:  {X_test.shape[0]} sequences ({y_test.mean():.2%} anomalies)")

sequence_length = SEQUENCE_LENGTH
n_features = X_train.shape[2]


# ============================================================================
# 5. LSTM AUTOENCODER MODEL
# ============================================================================

class LSTMAutoencoder:
    """LSTM Autoencoder for temporal anomaly detection"""

    def __init__(self, sequence_length, n_features, latent_dim=64):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.model = None
        self.history = None

    def build_model(self):
        """Build LSTM Autoencoder architecture"""
        # Encoder
        encoder_input = layers.Input(shape=(self.sequence_length, self.n_features), name='encoder_input')

        # LSTM encoder layers
        x = layers.LSTM(128, return_sequences=True)(encoder_input)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64, return_sequences=False)(x)
        x = layers.Dropout(0.3)(x)

        # Latent representation
        encoded = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)

        # Decoder
        # Repeat the latent vector to match sequence length
        x = layers.RepeatVector(self.sequence_length)(encoded)

        # LSTM decoder layers
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)

        # Output layer (reconstruct input)
        decoded = layers.TimeDistributed(layers.Dense(self.n_features), name='decoder_output')(x)

        # Full autoencoder
        self.model = models.Model(encoder_input, decoded, name='lstm_autoencoder')
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )

        return self.model

    def train(self, X_train, X_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the LSTM autoencoder"""
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
        # Mean squared error across all timesteps and features
        reconstruction_error = np.mean(np.square(X - X_reconstructed), axis=(1, 2))
        return reconstruction_error


# ============================================================================
# 6. LSTM BINARY CLASSIFIER MODEL
# ============================================================================

class LSTMClassifier:
    """LSTM Binary Classifier for temporal anomaly detection"""

    def __init__(self, sequence_length, n_features):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None

    def build_model(self):
        """Build LSTM Classifier architecture"""
        model_input = layers.Input(shape=(self.sequence_length, self.n_features), name='input')

        # LSTM layers
        x = layers.LSTM(128, return_sequences=True)(model_input)
        x = layers.Dropout(0.4)(x)
        x = layers.LSTM(64, return_sequences=False)(x)
        x = layers.Dropout(0.4)(x)

        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Output layer (binary classification)
        output = layers.Dense(1, activation='sigmoid', name='output')(x)

        # Full model
        self.model = models.Model(model_input, output, name='lstm_classifier')
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        return self.model

    def train(self, X_train, y_train, X_val, y_val, class_weights, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the LSTM classifier"""
        if self.model is None:
            self.build_model()

        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_auc',
            patience=PATIENCE,
            mode='max',
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

        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=[early_stop, reduce_lr],
            verbose=2
        )

        return self.history


# ============================================================================
# 7. TRAIN LSTM AUTOENCODER
# ============================================================================

print("\n" + "="*80)
print("TRAINING LSTM AUTOENCODER")
print("="*80)

lstm_ae = LSTMAutoencoder(sequence_length=sequence_length, n_features=n_features)
lstm_ae.build_model()
print(f"\nLSTM Autoencoder architecture:")
lstm_ae.model.summary()

print(f"\nTraining LSTM Autoencoder...")
start_time = datetime.now()
lstm_ae_history = lstm_ae.train(X_train, X_val)
lstm_ae_train_time = (datetime.now() - start_time).total_seconds()
print(f"Training completed in {lstm_ae_train_time:.1f} seconds")


# ============================================================================
# 8. TRAIN LSTM CLASSIFIER
# ============================================================================

print("\n" + "="*80)
print("TRAINING LSTM CLASSIFIER")
print("="*80)

# Calculate class weights for imbalanced data
class_counts = np.bincount(y_train.astype(int))
class_weights = {
    0: len(y_train) / (2 * class_counts[0]),
    1: len(y_train) / (2 * class_counts[1])
}
print(f"\nClass weights: {class_weights}")

lstm_clf = LSTMClassifier(sequence_length=sequence_length, n_features=n_features)
lstm_clf.build_model()
print(f"\nLSTM Classifier architecture:")
lstm_clf.model.summary()

print(f"\nTraining LSTM Classifier...")
start_time = datetime.now()
lstm_clf_history = lstm_clf.train(X_train, y_train, X_val, y_val, class_weights)
lstm_clf_train_time = (datetime.now() - start_time).total_seconds()
print(f"Training completed in {lstm_clf_train_time:.1f} seconds")


# ============================================================================
# 9. EVALUATE MODELS
# ============================================================================

print("\n" + "="*80)
print("EVALUATING MODELS")
print("="*80)

# LSTM Autoencoder - Compute reconstruction errors
print("\nComputing LSTM Autoencoder reconstruction errors...")
lstm_ae_train_errors = lstm_ae.predict_reconstruction_error(X_train)
lstm_ae_val_errors = lstm_ae.predict_reconstruction_error(X_val)
lstm_ae_test_errors = lstm_ae.predict_reconstruction_error(X_test)

# Compute ROC AUC scores (higher reconstruction error = anomaly)
lstm_ae_train_auc = roc_auc_score(y_train, lstm_ae_train_errors)
lstm_ae_val_auc = roc_auc_score(y_val, lstm_ae_val_errors)
lstm_ae_test_auc = roc_auc_score(y_test, lstm_ae_test_errors)

print(f"\nLSTM Autoencoder ROC AUC:")
print(f"  Train: {lstm_ae_train_auc:.4f}")
print(f"  Val:   {lstm_ae_val_auc:.4f}")
print(f"  Test:  {lstm_ae_test_auc:.4f}")

# Compute precision-recall AUC
lstm_ae_precision, lstm_ae_recall, _ = precision_recall_curve(y_test, lstm_ae_test_errors)
lstm_ae_pr_auc = auc(lstm_ae_recall, lstm_ae_precision)
print(f"  PR AUC: {lstm_ae_pr_auc:.4f}")

# LSTM Classifier - Get predictions
print("\nComputing LSTM Classifier predictions...")
lstm_clf_train_pred = lstm_clf.model.predict(X_train, verbose=0).flatten()
lstm_clf_val_pred = lstm_clf.model.predict(X_val, verbose=0).flatten()
lstm_clf_test_pred = lstm_clf.model.predict(X_test, verbose=0).flatten()

# Compute ROC AUC scores
lstm_clf_train_auc = roc_auc_score(y_train, lstm_clf_train_pred)
lstm_clf_val_auc = roc_auc_score(y_val, lstm_clf_val_pred)
lstm_clf_test_auc = roc_auc_score(y_test, lstm_clf_test_pred)

print(f"\nLSTM Classifier ROC AUC:")
print(f"  Train: {lstm_clf_train_auc:.4f}")
print(f"  Val:   {lstm_clf_val_auc:.4f}")
print(f"  Test:  {lstm_clf_test_auc:.4f}")

# Compute other metrics
lstm_clf_test_acc = accuracy_score(y_test, (lstm_clf_test_pred > 0.5).astype(int))
lstm_clf_test_f1 = f1_score(y_test, (lstm_clf_test_pred > 0.5).astype(int))

lstm_clf_precision, lstm_clf_recall, _ = precision_recall_curve(y_test, lstm_clf_test_pred)
lstm_clf_pr_auc = auc(lstm_clf_recall, lstm_clf_precision)

print(f"  Accuracy: {lstm_clf_test_acc:.4f}")
print(f"  F1 Score: {lstm_clf_test_f1:.4f}")
print(f"  PR AUC:   {lstm_clf_pr_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, (lstm_clf_test_pred > 0.5).astype(int))
print(f"\nConfusion Matrix:")
print(cm)


# ============================================================================
# 10. SAVE MODELS AND RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODELS AND RESULTS")
print("="*80)

# Save Keras models
lstm_ae_model_path = OUTPUT_DIR / "lstm_autoencoder_model.keras"
lstm_clf_model_path = OUTPUT_DIR / "lstm_classifier_model.keras"

lstm_ae.model.save(lstm_ae_model_path)
lstm_clf.model.save(lstm_clf_model_path)
print(f"Saved LSTM Autoencoder to: {lstm_ae_model_path}")
print(f"Saved LSTM Classifier to: {lstm_clf_model_path}")

# Save results
results = {
    'lstm_autoencoder': {
        'model_path': str(lstm_ae_model_path),
        'train_time': lstm_ae_train_time,
        'latent_dim': lstm_ae.latent_dim,
        'train_errors': lstm_ae_train_errors,
        'val_errors': lstm_ae_val_errors,
        'test_errors': lstm_ae_test_errors,
        'train_auc': lstm_ae_train_auc,
        'val_auc': lstm_ae_val_auc,
        'test_auc': lstm_ae_test_auc,
        'pr_auc': lstm_ae_pr_auc,
        'history': lstm_ae_history.history
    },
    'lstm_classifier': {
        'model_path': str(lstm_clf_model_path),
        'train_time': lstm_clf_train_time,
        'train_predictions': lstm_clf_train_pred,
        'val_predictions': lstm_clf_val_pred,
        'test_predictions': lstm_clf_test_pred,
        'train_auc': lstm_clf_train_auc,
        'val_auc': lstm_clf_val_auc,
        'test_auc': lstm_clf_test_auc,
        'test_accuracy': lstm_clf_test_acc,
        'test_f1': lstm_clf_test_f1,
        'pr_auc': lstm_clf_pr_auc,
        'history': lstm_clf_history.history
    },
    'data_splits': {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'train_anomaly_rate': float(y_train.mean()),
        'val_anomaly_rate': float(y_val.mean()),
        'test_anomaly_rate': float(y_test.mean()),
        'sequence_length': sequence_length,
        'n_features': n_features,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'block_size': BLOCK_SIZE,
    },
    'labels': {
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
}

results_path = OUTPUT_DIR / "lstm_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"Saved results to: {results_path}")


# ============================================================================
# 11. GENERATE PLOTS
# ============================================================================

print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

# Plot 1: Training history
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# LSTM Autoencoder - Loss
axes[0, 0].plot(lstm_ae_history.history['loss'], label='Train Loss', alpha=0.8)
axes[0, 0].plot(lstm_ae_history.history['val_loss'], label='Val Loss', alpha=0.8)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('LSTM Autoencoder - Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# LSTM Autoencoder - MAE
axes[0, 1].plot(lstm_ae_history.history['mae'], label='Train MAE', alpha=0.8)
axes[0, 1].plot(lstm_ae_history.history['val_mae'], label='Val MAE', alpha=0.8)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].set_title('LSTM Autoencoder - MAE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# LSTM Classifier - Loss
axes[1, 0].plot(lstm_clf_history.history['loss'], label='Train Loss', alpha=0.8)
axes[1, 0].plot(lstm_clf_history.history['val_loss'], label='Val Loss', alpha=0.8)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('LSTM Classifier - Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# LSTM Classifier - AUC
axes[1, 1].plot(lstm_clf_history.history['auc'], label='Train AUC', alpha=0.8)
axes[1, 1].plot(lstm_clf_history.history['val_auc'], label='Val AUC', alpha=0.8)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('AUC')
axes[1, 1].set_title('LSTM Classifier - AUC')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_history.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: training_history.png")

# Plot 2: Error/Prediction distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LSTM Autoencoder - Reconstruction errors
axes[0].hist(lstm_ae_test_errors[y_test == 0], bins=50, alpha=0.6, label='Normal', density=True)
axes[0].hist(lstm_ae_test_errors[y_test == 1], bins=50, alpha=0.6, label='Anomaly', density=True)
axes[0].set_xlabel('Reconstruction Error')
axes[0].set_ylabel('Density')
axes[0].set_title(f'LSTM Autoencoder - Test Set (AUC={lstm_ae_test_auc:.4f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# LSTM Classifier - Predictions
axes[1].hist(lstm_clf_test_pred[y_test == 0], bins=50, alpha=0.6, label='Normal', density=True)
axes[1].hist(lstm_clf_test_pred[y_test == 1], bins=50, alpha=0.6, label='Anomaly', density=True)
axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
axes[1].set_xlabel('Predicted Probability')
axes[1].set_ylabel('Density')
axes[1].set_title(f'LSTM Classifier - Test Set (AUC={lstm_clf_test_auc:.4f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "prediction_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: prediction_distributions.png")


print("\n" + "="*80)
print("STEP 05a COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nKey files generated:")
print(f"  - lstm_autoencoder_model.keras")
print(f"  - lstm_classifier_model.keras")
print(f"  - lstm_results.pkl")
print(f"  - training_history.png")
print(f"  - prediction_distributions.png")
print(f"\nBest model: {'LSTM Classifier' if lstm_clf_test_auc > lstm_ae_test_auc else 'LSTM Autoencoder'}")
print(f"Best Test AUC: {max(lstm_clf_test_auc, lstm_ae_test_auc):.4f}")
