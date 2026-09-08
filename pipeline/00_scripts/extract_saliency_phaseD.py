#!/usr/bin/env python3
"""Extract gradient-based saliency maps for a trained Phase D multi-head model.

Added 2026-09-05: Saliency was named in the pipeline description but not
implemented anywhere; none of the deep-learning models in this codebase
(04a, 05a, 06a, Phase D) had any explainability method at all -- only the
three classical multi-label models had SHAP.

Uses vanilla input-gradient saliency (Simonyan et al. 2013): for each sampled
event and each of the 7 fault-label output heads, computes
|d(sigmoid_output)/d(input)| via autograd. Simplest defensible saliency
method and the natural fit here since Phase D is already PyTorch (autograd
is free), unlike SmoothGrad/Integrated Gradients which need many forward
passes per sample -- a reasonable place to start, not a ceiling.

Usage:
    python extract_saliency_phaseD.py --arch cnn
"""
import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('extract_saliency_phaseD')

COOKED = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data'))
OUTDIR = COOKED / 'step_10_phaseD'

SALIENCY_SAMPLE_SIZE = 200  # cheap relative to SHAP/LIME: one backward pass per event, batched

# Import the encoder/head classes straight from phaseD_train.py rather than
# duplicating them by hand -- an earlier draft of this script hand-copied
# them and got MLPEncoder's architecture, layer names, and the Transformer's
# batch-first convention all wrong relative to the original, which would have
# made load_state_dict fail. Importing avoids that class of bug entirely.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from phaseD_train import CNNEncoder, MLPEncoder, CNNLSTMEncoder, TransformerEncoder, MultiHeadModel


def build_encoder(arch, n_features, bottleneck):
    if arch == 'cnn':
        return CNNEncoder(in_channels=n_features, bottleneck=bottleneck)
    if arch == 'transformer':
        return TransformerEncoder(input_dim=n_features, bottleneck=bottleneck)
    if arch == 'cnn_lstm':
        return CNNLSTMEncoder(in_channels=n_features, bottleneck=bottleneck)
    return MLPEncoder(input_dim=n_features, bottleneck=bottleneck)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=['cnn', 'transformer', 'cnn_lstm', 'mlp'], required=True)
    parser.add_argument('--bottleneck', type=int, default=256)
    args = parser.parse_args()

    ckpt_path = OUTDIR / args.arch / f'phaseD_best_{args.arch}.pth'
    if not ckpt_path.exists():
        logger.error('Checkpoint not found: %s (train phaseD_train.py --arch %s first)', ckpt_path, args.arch)
        raise SystemExit(1)

    logger.info('Loading checkpoint: %s', ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    feature_names = list(ckpt['feature_names'])
    scaler, pca = ckpt.get('scaler'), ckpt.get('pca')
    n_features = len(feature_names) if pca is None else pca.n_components_

    features_file = COOKED / 'features_engineered.pkl'
    logger.info('Loading features from %s', features_file)
    with open(features_file, 'rb') as f:
        pkl_data = pickle.load(f)
    y = pkl_data['y_multilabel'].astype(np.float32)
    n_labels = y.shape[1]

    X_raw = pkl_data['features_all'][feature_names].fillna(0).values
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e10, neginf=-1e10)
    X = scaler.transform(X_raw) if scaler is not None else X_raw
    if pca is not None:
        X = pca.transform(X)
    X = X.astype(np.float32)

    rng = np.random.RandomState(42)
    n_samples = min(SALIENCY_SAMPLE_SIZE, X.shape[0])
    sample_idx = rng.choice(X.shape[0], size=n_samples, replace=False)
    X_sample = torch.from_numpy(X[sample_idx])

    logger.info('Building %s encoder (n_features=%d, n_labels=%d, bottleneck=%d)',
                args.arch, n_features, n_labels, args.bottleneck)
    encoder = build_encoder(args.arch, n_features, args.bottleneck)
    model = MultiHeadModel(encoder, bottleneck=args.bottleneck, n_labels=n_labels)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    logger.info('Computing input-gradient saliency for %d samples x %d labels', n_samples, n_labels)
    X_sample.requires_grad_(True)
    # Phase 4 (2026-09-06): MultiHeadModel.forward() now returns raw logits
    # (needed for BCEWithLogitsLoss(pos_weight=...) during training), not
    # sigmoid probabilities -- apply sigmoid explicitly here so this script's
    # documented "d(sigmoid_output)/d(input)" saliency semantics are unchanged.
    outputs = torch.sigmoid(model(X_sample))  # (n_samples, n_labels)

    saliency_dict = {}
    for label_idx in range(n_labels):
        if X_sample.grad is not None:
            X_sample.grad.zero_()
        # Sum over the batch so backward() gives, per input row, the gradient
        # of THAT row's own output for this label (batched rows don't
        # interact through the encoder here, so this is exact, not an
        # approximation).
        outputs[:, label_idx].sum().backward(retain_graph=True)
        saliency = X_sample.grad.detach().abs().numpy()
        saliency_dict[f'saliency_{label_idx}'] = saliency.astype(np.float32)
        logger.info('  Label %d: saliency shape = %s', label_idx, saliency.shape)

    saliency_dict['sample_idx'] = sample_idx
    saliency_dict['feature_names'] = np.array(feature_names if pca is None else
                                               [f'pc_{i}' for i in range(n_features)], dtype=object)

    out_path = OUTDIR / args.arch / f'saliency_{args.arch}.npz'
    np.savez_compressed(out_path, **saliency_dict)
    file_size = out_path.stat().st_size / (1024 * 1024)
    logger.info('Saved: %s (%.2f MB)', out_path, file_size)


if __name__ == '__main__':
    main()
