#!/usr/bin/env python3
"""Phase D training: Multi-head deep models (PyTorch).

Supports architectures: cnn, transformer, cnn_lstm, mlp.
Loads features from the cooked data artifact and trains a multi-head model
producing per-label binary outputs. Saves checkpoint and summary.
"""
import argparse
import logging
import sys
from pathlib import Path
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import hamming_loss, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from leakage_safe_features import leakage_safe_split
from split_diagnostics import log_split_composition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('phaseD')

import os
COOKED = Path(os.environ.get('SPIRAL2_COOKED_DIR', '/sps/m4cast/_spiral2_data/_llrf_data/cooked_data'))
OUTDIR = COOKED / 'step_10_phaseD'
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_features():
    candidates = [COOKED / 'step_03_features' / 'features_engineered.pkl', COOKED / 'features_engineered.pkl']
    for p in candidates:
        if p.exists():
            logger.info('Loading features from %s', p)
            with open(p, 'rb') as f:
                obj = pickle.load(f)
            y = obj.get('y_multilabel')
            return obj, y, obj.get('feature_cols')
    logger.info('No features artifact found; creating synthetic data')
    rng = np.random.RandomState(0)
    X = rng.normal(size=(1000, 128)).astype(np.float32)
    y = (rng.rand(1000, 7) > 0.95).astype(np.float32)
    feature_names = [f'feat_{i}' for i in range(X.shape[1])]
    return {'synthetic_X': X}, y, feature_names


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, bottleneck=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, bottleneck)

    def forward(self, x):
        # x: (batch, features) -> treat features as channels x length=1
        x = x.unsqueeze(-1)  # (B, 1243, 1) - features as channels
        h = self.net(x).squeeze(-1)
        return self.fc(h)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, bottleneck=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, bottleneck),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CNNLSTMEncoder(nn.Module):
    def __init__(self, in_channels, bottleneck=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, bottleneck)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (B, 1243, 1) - features as channels
        h = self.conv(x).permute(0, 2, 1)  # (B, L, C)
        out, _ = self.lstm(h)
        return self.fc(out.mean(dim=1))


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=4, bottleneck=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, bottleneck)

    def forward(self, x):
        # x: (B, input_dim) -> treat as sequence length=1 of embeddings
        x = x.unsqueeze(1)  # (B, L=1, D_in)
        x = self.proj(x)
        x = x.permute(1, 0, 2)  # (L, B, D)
        h = self.encoder(x)
        h = h.mean(dim=0)
        return self.fc(h)


class MultiHeadModel(nn.Module):
    def __init__(self, encoder, bottleneck, n_labels):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(bottleneck, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)) for _ in range(n_labels)])

    def forward(self, x):
        # Phase 4 (2026-09-06): returns raw logits, not sigmoid probabilities --
        # needed so BCEWithLogitsLoss(pos_weight=...) can be used for per-label
        # class weighting (plain BCELoss has no pos_weight support since it
        # operates on already-squashed [0,1] probabilities). Callers that need
        # probabilities (eval_model(), inference) apply torch.sigmoid explicitly.
        h = self.encoder(x)
        outs = [head(h).squeeze(-1) for head in self.heads]
        return torch.stack(outs, dim=1)


def train_epoch(model, dl, opt, loss_fn, device):
    model.train()
    tot_loss = 0.0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        opt.step()
        tot_loss += float(loss.item()) * xb.size(0)
    return tot_loss / len(dl.dataset)


def eval_model(model, dl, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            out = torch.sigmoid(model(xb)).cpu().numpy()  # model returns logits; sigmoid -> probabilities
            ys.append(yb.numpy())
            ps.append(out)
    y = np.vstack(ys)
    p = np.vstack(ps)
    p_bin = (p >= 0.5).astype(int)
    ham = hamming_loss(y, p_bin)
    micro = f1_score(y, p_bin, average='micro', zero_division=0)
    macro = f1_score(y, p_bin, average='macro', zero_division=0)
    return {'hamming': float(ham), 'micro_f1': float(micro), 'macro_f1': float(macro)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=['cnn', 'transformer', 'cnn_lstm', 'mlp'], default='cnn')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--bottleneck', type=int, default=256)
    args = parser.parse_args()

    pkl_data, y, feature_names = load_features()
    if y is None:
        logger.error('Missing data')
        return
    y = y.astype(np.float32)
    n_labels = y.shape[1]

    # train/val split (leakage-safe: scaler fit on the train fold only, NOT
    # on the globally pre-fit X_scaled -- the original code's 80/20 split had
    # no held-out test set beyond this val split, so this is unchanged here).
    fit_scaler, fit_pca = None, None
    if 'synthetic_X' in pkl_data:
        X = pkl_data['synthetic_X'].astype(np.float32)
        idx = np.arange(X.shape[0])
        np.random.seed(0)
        np.random.shuffle(idx)
        split_i = int(0.8 * len(idx))
        tr, va = idx[:split_i], idx[split_i:]
        X_tr, X_va, y_tr, y_va = X[tr], X[va], y[tr], y[va]
    else:
        split = leakage_safe_split(pkl_data, y, test_size=0.2, random_state=0,
                                    stratify=False)
        X_tr, X_va = split['X_train'].astype(np.float32), split['X_test'].astype(np.float32)
        y_tr, y_va = split['y_train'], split['y_test']
        fit_scaler, fit_pca = split['scaler'], split['pca']
        label_names_d = pkl_data.get('fault_column_names') or [f'label_{i}' for i in range(y.shape[1])]
        log_split_composition(y_tr, y_va, label_names_d, logger,
                               title=f'Phase D ({args.arch}) train/val composition')

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    logger.info('Building model: %s', args.arch)
    n_features = X_tr.shape[1]
    if args.arch == 'cnn':
        enc = CNNEncoder(in_channels=n_features, bottleneck=args.bottleneck)
    elif args.arch == 'transformer':
        enc = TransformerEncoder(input_dim=n_features, bottleneck=args.bottleneck)
    elif args.arch == 'cnn_lstm':
        enc = CNNLSTMEncoder(in_channels=n_features, bottleneck=args.bottleneck)
    else:
        enc = MLPEncoder(input_dim=n_features, bottleneck=args.bottleneck)

    model = MultiHeadModel(enc, bottleneck=args.bottleneck, n_labels=n_labels).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # Phase 4 (2026-09-06): per-label pos_weight (neg_count/pos_count in the
    # TRAIN fold only), computed once here and passed to BCEWithLogitsLoss --
    # unlike XGBoost's scale_pos_weight (one value per whole model), PyTorch's
    # pos_weight is a proper per-label vector, so this is genuinely per-label
    # balanced, not an aggregate approximation.
    pos_counts = y_tr.sum(axis=0)
    neg_counts = y_tr.shape[0] - pos_counts
    pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32).to(device)
    logger.info('Per-label pos_weight (train fold): %s', np.round(pos_weight.cpu().numpy(), 2).tolist())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_micro = -1.0
    summary = {'config': vars(args)}
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, tr_dl, opt, loss_fn, device)
        metrics = eval_model(model, va_dl, device)
        logger.info('Epoch %d train_loss=%.4f val_micro=%.4f val_macro=%.4f val_hamming=%.6f (%.1fs)', ep, train_loss, metrics['micro_f1'], metrics['macro_f1'], metrics['hamming'], time.time()-t0)
        if metrics['micro_f1'] > best_micro:
            best_micro = metrics['micro_f1']
            arch_dir = OUTDIR / args.arch
            arch_dir.mkdir(parents=True, exist_ok=True)
            ckpt = arch_dir / f'phaseD_best_{args.arch}.pth'
            torch.save({
                'model_state': model.state_dict(), 'arch': args.arch, 'feature_names': feature_names,
                'scaler': fit_scaler, 'pca': fit_pca,
            }, ckpt)
            summary['best'] = {'epoch': ep, 'metrics': metrics}

    # final save
    summary['final'] = metrics
    arch_dir = OUTDIR / args.arch
    arch_dir.mkdir(parents=True, exist_ok=True)
    summary_path = arch_dir / f'phaseD_summary_{args.arch}.pkl'
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    logger.info('Saved summary to %s', summary_path)


if __name__ == '__main__':
    main()
