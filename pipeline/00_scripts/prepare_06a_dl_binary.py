#!/usr/bin/env python3
"""
Step 06a: Deep Learning Binary Classification
----------------------------------------------
Implements Deep Neural Network (DNN) and 1D Convolutional Neural Network (CNN)
for binary classification of LLRF anomalies.

This script compares DL approaches against classical ML (RF, XGBoost) from Step 06.

Output: Trained models and predictions saved to step_06a_dl_binary/
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
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score
)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import leakage_safe_split


print("="*80)
print("STEP 06a: DEEP LEARNING BINARY CLASSIFICATION")
print("="*80)
print(f"\nTensorFlow version: {tf.__version__}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}\n")


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

import os
DATA_DIR = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data'))
OUTPUT_DIR = DATA_DIR / "step_06a_dl_binary"
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
print(f"Class distribution: Normal={np.sum(y_binary==0)}, Anomaly={np.sum(y_binary==1)}")


# ============================================================================
# 3. PREPARE TRAINING DATA
# ============================================================================

# Leakage-safe train/val/test split: scaler fit on the TRAIN fold only, not
# on the globally pre-fit X_scaled. test_size 0.2 of the total + val_size
# 0.16 of the total reproduces the original 64/16/20 split (0.2 test, then
# 0.2 of the remaining 0.8 for val).
_split = leakage_safe_split(data, y_binary, test_size=0.2, val_size=0.16,
                             random_state=RANDOM_SEED)
X_train, X_val, X_test = _split['X_train'], _split['X_val'], _split['X_test']
y_train, y_val, y_test = _split['y_train'], _split['y_val'], _split['y_test']

print(f"\nData splits:")
print(f"  Train: {X_train.shape[0]} events ({y_train.mean():.2%} anomalies)")
print(f"  Val:   {X_val.shape[0]} events ({y_val.mean():.2%} anomalies)")
print(f"  Test:  {X_test.shape[0]} events ({y_test.mean():.2%} anomalies)")

input_dim = X_train.shape[1]

# Compute class weights for imbalanced data
class_weights = {
    0: len(y_train) / (2 * np.sum(y_train == 0)),
    1: len(y_train) / (2 * np.sum(y_train == 1))
}
print(f"\nClass weights: {class_weights}")


# ============================================================================
# 4. DEEP NEURAL NETWORK (DNN) MODEL
# ============================================================================

class DNN:
    """Deep Neural Network for binary classification"""

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None
        self.history = None

    def build_model(self):
        """Build DNN architecture"""
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),

            # Layer 1
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            # Layer 2
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            # Layer 3
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Layer 4
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Output layer
            layers.Dense(1, activation='sigmoid')
        ], name='DNN')

        self.model = model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )

        return self.model

    def train(self, X_train, y_train, X_val, y_val, class_weights,
              epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the DNN"""
        if self.model is None:
            self.build_model()

        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_auc',
            patience=PATIENCE,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='max',
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

    def predict(self, X):
        """Predict probabilities"""
        return self.model.predict(X, verbose=0).flatten()


# ============================================================================
# 5. 1D CONVOLUTIONAL NEURAL NETWORK (CNN) MODEL
# ============================================================================

class CNN1D:
    """1D Convolutional Neural Network for binary classification"""

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None
        self.history = None

    def build_model(self):
        """Build 1D CNN architecture"""
        model_input = layers.Input(shape=(self.input_dim, 1))

        # Conv Block 1
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(model_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.3)(x)

        # Conv Block 2
        x = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.3)(x)

        # Conv Block 3
        x = layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.4)(x)

        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Output
        output = layers.Dense(1, activation='sigmoid')(x)

        self.model = models.Model(model_input, output, name='CNN1D')
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )

        return self.model

    def train(self, X_train, y_train, X_val, y_val, class_weights,
              epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the CNN"""
        if self.model is None:
            self.build_model()

        # Reshape for CNN (add channel dimension)
        X_train_cnn = X_train.reshape(-1, self.input_dim, 1)
        X_val_cnn = X_val.reshape(-1, self.input_dim, 1)

        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_auc',
            patience=PATIENCE,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='max',
            verbose=1
        )

        # Train
        self.history = self.model.fit(
            X_train_cnn, y_train,
            validation_data=(X_val_cnn, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=[early_stop, reduce_lr],
            verbose=2
        )

        return self.history

    def predict(self, X):
        """Predict probabilities"""
        X_cnn = X.reshape(-1, self.input_dim, 1)
        return self.model.predict(X_cnn, verbose=0).flatten()


# ============================================================================
# 6. TRAIN MODELS
# ============================================================================

print("\n" + "="*80)
print("TRAINING DEEP NEURAL NETWORK (DNN)")
print("="*80)

dnn = DNN(input_dim=input_dim)
dnn.build_model()
print(f"\nDNN architecture:")
dnn.model.summary()

print(f"\nTraining DNN...")
start_time = datetime.now()
dnn_history = dnn.train(X_train, y_train, X_val, y_val, class_weights)
dnn_train_time = (datetime.now() - start_time).total_seconds()
print(f"Training completed in {dnn_train_time:.1f} seconds")


print("\n" + "="*80)
print("TRAINING 1D CONVOLUTIONAL NEURAL NETWORK (CNN)")
print("="*80)

cnn = CNN1D(input_dim=input_dim)
cnn.build_model()
print(f"\nCNN architecture:")
cnn.model.summary()

print(f"\nTraining CNN...")
start_time = datetime.now()
cnn_history = cnn.train(X_train, y_train, X_val, y_val, class_weights)
cnn_train_time = (datetime.now() - start_time).total_seconds()
print(f"Training completed in {cnn_train_time:.1f} seconds")


# ============================================================================
# 7. EVALUATE MODELS
# ============================================================================

print("\n" + "="*80)
print("EVALUATING MODELS")
print("="*80)

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    """Comprehensive evaluation of a model"""
    results = {}

    # Predict probabilities
    train_proba = model.predict(X_train)
    val_proba = model.predict(X_val)
    test_proba = model.predict(X_test)

    # Predict classes (threshold = 0.5)
    train_pred = (train_proba >= 0.5).astype(int)
    val_pred = (val_proba >= 0.5).astype(int)
    test_pred = (test_proba >= 0.5).astype(int)

    # Metrics for each split
    for split_name, y_true, y_pred, y_proba in [
        ('train', y_train, train_pred, train_proba),
        ('val', y_val, val_pred, val_proba),
        ('test', y_test, test_pred, test_proba)
    ]:
        results[f'{split_name}_accuracy'] = accuracy_score(y_true, y_pred)
        results[f'{split_name}_precision'] = precision_score(y_true, y_pred)
        results[f'{split_name}_recall'] = recall_score(y_true, y_pred)
        results[f'{split_name}_f1'] = f1_score(y_true, y_pred)
        results[f'{split_name}_roc_auc'] = roc_auc_score(y_true, y_proba)

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        results[f'{split_name}_pr_auc'] = auc(recall, precision)

    # Store predictions
    results['train_proba'] = train_proba
    results['val_proba'] = val_proba
    results['test_proba'] = test_proba
    results['train_pred'] = train_pred
    results['val_pred'] = val_pred
    results['test_pred'] = test_pred

    # Print results
    print(f"\n{model_name} Performance:")
    print(f"  Train: Acc={results['train_accuracy']:.4f}, Prec={results['train_precision']:.4f}, "
          f"Rec={results['train_recall']:.4f}, F1={results['train_f1']:.4f}, "
          f"ROC_AUC={results['train_roc_auc']:.4f}")
    print(f"  Val:   Acc={results['val_accuracy']:.4f}, Prec={results['val_precision']:.4f}, "
          f"Rec={results['val_recall']:.4f}, F1={results['val_f1']:.4f}, "
          f"ROC_AUC={results['val_roc_auc']:.4f}")
    print(f"  Test:  Acc={results['test_accuracy']:.4f}, Prec={results['test_precision']:.4f}, "
          f"Rec={results['test_recall']:.4f}, F1={results['test_f1']:.4f}, "
          f"ROC_AUC={results['test_roc_auc']:.4f}")

    return results


dnn_results = evaluate_model(dnn, X_train, y_train, X_val, y_val, X_test, y_test, "DNN")
cnn_results = evaluate_model(cnn, X_train, y_train, X_val, y_val, X_test, y_test, "CNN")


# Ensemble predictions (average probabilities)
print("\n" + "="*80)
print("ENSEMBLE MODEL (DNN + CNN)")
print("="*80)

ensemble_train_proba = (dnn_results['train_proba'] + cnn_results['train_proba']) / 2
ensemble_val_proba = (dnn_results['val_proba'] + cnn_results['val_proba']) / 2
ensemble_test_proba = (dnn_results['test_proba'] + cnn_results['test_proba']) / 2

ensemble_train_pred = (ensemble_train_proba >= 0.5).astype(int)
ensemble_val_pred = (ensemble_val_proba >= 0.5).astype(int)
ensemble_test_pred = (ensemble_test_proba >= 0.5).astype(int)

ensemble_results = {
    'train_accuracy': accuracy_score(y_train, ensemble_train_pred),
    'train_precision': precision_score(y_train, ensemble_train_pred),
    'train_recall': recall_score(y_train, ensemble_train_pred),
    'train_f1': f1_score(y_train, ensemble_train_pred),
    'train_roc_auc': roc_auc_score(y_train, ensemble_train_proba),
    'val_accuracy': accuracy_score(y_val, ensemble_val_pred),
    'val_precision': precision_score(y_val, ensemble_val_pred),
    'val_recall': recall_score(y_val, ensemble_val_pred),
    'val_f1': f1_score(y_val, ensemble_val_pred),
    'val_roc_auc': roc_auc_score(y_val, ensemble_val_proba),
    'test_accuracy': accuracy_score(y_test, ensemble_test_pred),
    'test_precision': precision_score(y_test, ensemble_test_pred),
    'test_recall': recall_score(y_test, ensemble_test_pred),
    'test_f1': f1_score(y_test, ensemble_test_pred),
    'test_roc_auc': roc_auc_score(y_test, ensemble_test_proba),
    'train_proba': ensemble_train_proba,
    'val_proba': ensemble_val_proba,
    'test_proba': ensemble_test_proba,
    'train_pred': ensemble_train_pred,
    'val_pred': ensemble_val_pred,
    'test_pred': ensemble_test_pred
}

print(f"\nEnsemble Performance:")
print(f"  Train: Acc={ensemble_results['train_accuracy']:.4f}, Prec={ensemble_results['train_precision']:.4f}, "
      f"Rec={ensemble_results['train_recall']:.4f}, F1={ensemble_results['train_f1']:.4f}, "
      f"ROC_AUC={ensemble_results['train_roc_auc']:.4f}")
print(f"  Val:   Acc={ensemble_results['val_accuracy']:.4f}, Prec={ensemble_results['val_precision']:.4f}, "
      f"Rec={ensemble_results['val_recall']:.4f}, F1={ensemble_results['val_f1']:.4f}, "
      f"ROC_AUC={ensemble_results['val_roc_auc']:.4f}")
print(f"  Test:  Acc={ensemble_results['test_accuracy']:.4f}, Prec={ensemble_results['test_precision']:.4f}, "
      f"Rec={ensemble_results['test_recall']:.4f}, F1={ensemble_results['test_f1']:.4f}, "
      f"ROC_AUC={ensemble_results['test_roc_auc']:.4f}")


# ============================================================================
# 8. SAVE MODELS AND RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODELS AND RESULTS")
print("="*80)

# Save Keras models
dnn_model_path = OUTPUT_DIR / "dnn_model.keras"
cnn_model_path = OUTPUT_DIR / "cnn_model.keras"

dnn.model.save(dnn_model_path)
cnn.model.save(cnn_model_path)
print(f"Saved DNN to: {dnn_model_path}")
print(f"Saved CNN to: {cnn_model_path}")

# Save results
results = {
    'dnn': {
        'model_path': str(dnn_model_path),
        'train_time': dnn_train_time,
        'history': dnn_history.history,
        **dnn_results
    },
    'cnn': {
        'model_path': str(cnn_model_path),
        'train_time': cnn_train_time,
        'history': cnn_history.history,
        **cnn_results
    },
    'ensemble': ensemble_results,
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

results_path = OUTPUT_DIR / "dl_binary_results.pkl"
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
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# DNN - Loss
axes[0, 0].plot(dnn_history.history['loss'], label='Train Loss', alpha=0.8)
axes[0, 0].plot(dnn_history.history['val_loss'], label='Val Loss', alpha=0.8)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('DNN Training Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# DNN - AUC
axes[0, 1].plot(dnn_history.history['auc'], label='Train AUC', alpha=0.8)
axes[0, 1].plot(dnn_history.history['val_auc'], label='Val AUC', alpha=0.8)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('AUC')
axes[0, 1].set_title('DNN Training AUC')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# CNN - Loss
axes[1, 0].plot(cnn_history.history['loss'], label='Train Loss', alpha=0.8)
axes[1, 0].plot(cnn_history.history['val_loss'], label='Val Loss', alpha=0.8)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('CNN Training Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# CNN - AUC
axes[1, 1].plot(cnn_history.history['auc'], label='Train AUC', alpha=0.8)
axes[1, 1].plot(cnn_history.history['val_auc'], label='Val AUC', alpha=0.8)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('AUC')
axes[1, 1].set_title('CNN Training AUC')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_history.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: training_history.png")

# Plot 2: ROC curves
fig, ax = plt.subplots(figsize=(10, 8))

for model_name, results_dict in [('DNN', dnn_results), ('CNN', cnn_results), ('Ensemble', ensemble_results)]:
    fpr, tpr, _ = roc_curve(y_test, results_dict['test_proba'])
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={results_dict['test_roc_auc']:.4f})", linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - Test Set')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: roc_curves.png")

# Plot 3: Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (model_name, results_dict) in enumerate([('DNN', dnn_results), ('CNN', cnn_results), ('Ensemble', ensemble_results)]):
    cm = confusion_matrix(y_test, results_dict['test_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('True')
    axes[idx].set_title(f'{model_name} - Confusion Matrix\n(Acc={results_dict["test_accuracy"]:.3f})')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrices.png")


print("\n" + "="*80)
print("STEP 06a COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nKey files generated:")
print(f"  - dnn_model.keras")
print(f"  - cnn_model.keras")
print(f"  - dl_binary_results.pkl")
print(f"  - training_history.png")
print(f"  - roc_curves.png")
print(f"  - confusion_matrices.png")
print(f"\nBest model: Ensemble")
print(f"Best Test ROC AUC: {ensemble_results['test_roc_auc']:.4f}")
print(f"Best Test F1: {ensemble_results['test_f1']:.4f}")
