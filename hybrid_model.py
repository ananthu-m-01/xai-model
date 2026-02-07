from __future__ import annotations

import os
import json
import time
import logging
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("multimodal_eeg_fmri_text")

# =========================
# Config
# =========================
class Config:
    # Repro
    SEED = 42

    # Model dims
    DROPOUT = 0.2
    HIDDEN_DIM = 64
    NUM_HEADS = 2
    FUSION_HIDDEN_DIM = 128

    # Train
    EPOCHS = 30
    LEARNING_RATE_BASE = 1e-4
    LEARNING_RATE_MAX = 5e-4
    WEIGHT_DECAY = 5e-3
    BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    TEST_BATCH_SIZE = 8
    PATIENCE = 5
    GRADIENT_ACCUMULATION_STEPS = 2

    # Data
    NUM_AUGMENTATIONS = 2

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Text
    TRANSFORMER_MODEL = "distilbert-base-uncased"
    MAX_TEXT_LENGTH = 256
    TEXT_HIDDEN_DIM = 768
    TEXT_FEATURES = [
        'dementia_history_parents',
        'learning_deficits',
        'other_diseases'
    ]
    UNFREEZE_LAST_N = 2  # selective unfreezing

    # Memory and speed
    USE_GRADIENT_CHECKPOINTING = True
    USE_TORCH_COMPILE = False  # was True

    # Loss / imbalance
    FOCAL_GAMMA = 2.0
    USE_FOCAL = True  # set False to use CE
    ALPHA_POS = 1.0   # will be set from data if None

    # Threshold tuning / calibration
    USE_CALIBRATION = False  # if True, Platt scaling (logistic) via CV
    N_SPLITS_CAL = 3


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Dataset
# =========================
class BrainDataset(Dataset):
    def __init__(self, eeg_features: np.ndarray, fmri_features: np.ndarray,
                 text_data: pd.DataFrame, labels: np.ndarray,
                 augment: bool = False, num_augmentations: int = 2):
        assert len(eeg_features) == len(fmri_features) == len(labels) == len(text_data)
        self.original_eeg = torch.as_tensor(eeg_features, dtype=torch.float32)
        self.original_fmri = torch.as_tensor(fmri_features, dtype=torch.float32)
        self.original_labels = torch.as_tensor(labels, dtype=torch.long)
        self.text_data = text_data.reset_index(drop=True)
        self.augment = augment
        self.num_augmentations = num_augmentations

        self.tokenizer = AutoTokenizer.from_pretrained(Config.TRANSFORMER_MODEL)
        self.processed_text = self._process_text()

        if augment:
            self.eeg, self.fmri, self.labels = self._augment()
        else:
            self.eeg = self.original_eeg
            self.fmri = self.original_fmri
            self.labels = self.original_labels

    def _process_text(self) -> list:
        out = []
        for _, row in self.text_data.iterrows():
            parts = []
            for f in Config.TEXT_FEATURES:
                val = row.get(f, "")
                if pd.notna(val) and str(val).strip():
                    parts.append(f"{f}: {val}")
            text = " [SEP] ".join(parts) if parts else "No data"
            enc = self.tokenizer(
                text, max_length=Config.MAX_TEXT_LENGTH, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            out.append({
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0)
            })
        return out

    def _augment(self):
        eegs = [self.original_eeg]
        fmr = [self.original_fmri]
        labs = [self.original_labels]
        for _ in range(self.num_augmentations):
            eeg_noise = self.original_eeg + torch.randn_like(self.original_eeg) * 0.05
            fmri_noise = self.original_fmri + torch.randn_like(self.original_fmri) * 0.05
            eegs.append(eeg_noise)
            fmr.append(fmri_noise)
            labs.append(self.original_labels)
        return torch.cat(eegs, 0), torch.cat(fmr, 0), torch.cat(labs, 0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        text_idx = idx % len(self.original_labels) if self.augment else idx
        text = self.processed_text[text_idx]
        text_inputs = {
            'input_ids': text['input_ids'],
            'attention_mask': text['attention_mask']
        }
        return self.eeg[idx], self.fmri[idx], text_inputs, self.labels[idx]


# =========================
# Model
# =========================
class EfficientAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MultiModalNet(nn.Module):
    def __init__(self, eeg_dim: int, fmri_dim: int):
        super().__init__()
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_dim, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
        )
        self.fmri_encoder = nn.Sequential(
            nn.Linear(fmri_dim, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
        )

        self.text_encoder = AutoModel.from_pretrained(Config.TRANSFORMER_MODEL)
        self.text_projector = nn.Linear(Config.TEXT_HIDDEN_DIM, Config.HIDDEN_DIM)

        self.cross_attn = EfficientAttention(Config.HIDDEN_DIM, Config.NUM_HEADS, Config.DROPOUT)

        self.fusion = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(Config.FUSION_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT)
        )

        # Main classifier (fused)
        self.classifier = nn.Linear(Config.FUSION_HIDDEN_DIM, 2)

        # Per-modality heads for decision-level fusion/stacking
        self.head_eeg = nn.Linear(Config.HIDDEN_DIM, 2)
        self.head_fmri = nn.Linear(Config.HIDDEN_DIM, 2)
        self.head_text = nn.Linear(Config.HIDDEN_DIM, 2)

        self._init_weights()

        # Selective unfreezing
        self._set_transformer_trainable(unfreeze_last_n=Config.UNFREEZE_LAST_N)

        # Optional torch.compile with safe fallback on Windows
        if Config.USE_TORCH_COMPILE and hasattr(torch, "compile"):
            import shutil, platform
            backend = "inductor"
            if os.name == "nt" and shutil.which("cl") is None:
                logger.warning("MSVC 'cl' not found; using torch.compile backend='aot_eager'.")
                backend = "aot_eager"
            try:
                self.forward = torch.compile(self.forward, backend=backend)
            except Exception as e:
                logger.warning(f"torch.compile disabled due to: {e}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _set_transformer_trainable(self, unfreeze_last_n=2):
        # DistilBERT has 6 transformer layers
        layers = list(self.text_encoder.transformer.layer)
        L = len(layers)
        for i, layer in enumerate(layers):
            req = (i >= L - unfreeze_last_n)
            for p in layer.parameters():
                p.requires_grad = req
        # Always train embeddings/projector?
        for p in self.text_encoder.embeddings.parameters():
            p.requires_grad = False

    def encode_text(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if Config.USE_GRADIENT_CHECKPOINTING and self.training:
            # Wrap forward for checkpoint to accept tensors only; use kwargs by closure
            def enc_fn(input_ids, attention_mask):
                return self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            last_hidden = checkpoint(enc_fn, text_inputs['input_ids'], text_inputs['attention_mask'])
        else:
            out = self.text_encoder(**text_inputs)
            # Prefer attribute access but be robust to different return types (dict/tuple/BaseModelOutput)
            last_hidden = getattr(out, "last_hidden_state", None)
            if last_hidden is None:
                # fallback to sequence-first element when last_hidden_state is not provided
                if isinstance(out, (tuple, list)) and len(out) > 0:
                    last_hidden = out[0]
                elif isinstance(out, dict):
                    last_hidden = out.get("last_hidden_state", None)
        if last_hidden is None:
            raise RuntimeError("Transformer did not return hidden states (last_hidden_state is None).")
        cls = last_hidden[:, 0, :]
        return self.text_projector(cls)

    def forward(self, eeg: torch.Tensor, fmri: torch.Tensor, text_inputs: Dict[str, torch.Tensor]):
        eeg_feat = self.eeg_encoder(eeg)
        fmri_feat = self.fmri_encoder(fmri)
        text_feat = self.encode_text(text_inputs)

        # Per-modality logits for optional stacking or auxiliary loss
        logits_eeg = self.head_eeg(eeg_feat)
        logits_fmri = self.head_fmri(fmri_feat)
        logits_text = self.head_text(text_feat)

        # Simple token axis for attention: [B, 2, C] using brain/text
        combined_tokens = torch.stack([0.5 * (eeg_feat + fmri_feat), text_feat], dim=1)
        attended = self.cross_attn(combined_tokens).mean(dim=1)

        fused = self.fusion(attended)
        logits = self.classifier(fused)
        return logits, logits_eeg, logits_fmri, logits_text


# =========================
# Losses and samplers
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = loss * self.alpha.gather(0, target)
        return loss.mean() if self.reduction == 'mean' else loss.sum()


def make_weighted_sampler(y_train: np.ndarray):
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_train].astype(float)
    return WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)


# =========================
# Optimizer param groups
# =========================
def build_param_groups_discriminative(model: MultiModalNet, base_lr=1e-4, max_lr=5e-4):
    params = []
    enc = model.text_encoder
    layers = list(enc.transformer.layer)
    L = len(layers)
    for i, layer in enumerate(layers):
        lr = base_lr + (max_lr - base_lr) * (i + 1) / max(1, L)
        params.append({'params': [p for p in layer.parameters() if p.requires_grad], 'lr': lr})

    # Other modules at higher LR
    high_lr_modules = [
        model.eeg_encoder, model.fmri_encoder,
        model.text_projector, model.fusion,
        model.classifier, model.head_eeg, model.head_fmri, model.head_text
    ]
    for m in high_lr_modules:
        params.append({'params': [p for p in m.parameters() if p.requires_grad], 'lr': max_lr})
    return params


# =========================
# Metrics helpers
# =========================
def tune_threshold_roc_gmean(y_true: np.ndarray, y_prob_pos: np.ndarray):
    fpr, tpr, thr = roc_curve(y_true, y_prob_pos)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = int(np.argmax(gmeans))
    return float(thr[ix]), float(gmeans[ix])


# =========================
# Training / Evaluation
# =========================
def train_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    running_loss = 0.0
    preds, trues = [], []
    use_amp = Config.DEVICE.type == 'cuda'
    accum = Config.GRADIENT_ACCUMULATION_STEPS
    optimizer.zero_grad(set_to_none=True)

    for step, (eeg, fmri, text_inputs, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        eeg = eeg.to(Config.DEVICE, non_blocking=True)
        fmri = fmri.to(Config.DEVICE, non_blocking=True)
        labels = labels.to(Config.DEVICE, non_blocking=True)
        text_inputs = {k: v.to(Config.DEVICE, non_blocking=True) for k, v in text_inputs.items()}

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, le, lf, lt = model(eeg, fmri, text_inputs)
            # Primary loss on fused logits + small auxiliary losses
            loss_main = criterion(logits, labels)
            aux = 0.2 * (criterion(le, labels) + criterion(lf, labels) + criterion(lt, labels)) / 3.0
            loss = (loss_main + aux) / accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * labels.size(0) * accum
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        trues.extend(labels.detach().cpu().numpy().tolist())

    dataset_size = max(1, len(trues))
    return {
        'loss': running_loss / dataset_size,
        'accuracy': float(np.mean(np.array(preds) == np.array(trues)))
    }


@torch.no_grad()
def evaluate(model, loader, criterion=None, return_probs=False):
    model.eval()
    running_loss = 0.0
    preds, trues = [], []
    probs = []

    for eeg, fmri, text_inputs, labels in tqdm(loader, desc="Eval", leave=False):
        eeg = eeg.to(Config.DEVICE, non_blocking=True)
        fmri = fmri.to(Config.DEVICE, non_blocking=True)
        labels = labels.to(Config.DEVICE, non_blocking=True)
        text_inputs = {k: v.to(Config.DEVICE, non_blocking=True) for k, v in text_inputs.items()}

        logits, _, _, _ = model(eeg, fmri, text_inputs)
        if criterion is not None:
            running_loss += criterion(logits, labels).item() * labels.size(0)
        p = F.softmax(logits, dim=1)[:, 1]
        probs.extend(p.detach().cpu().numpy().tolist())
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        trues.extend(labels.detach().cpu().numpy().tolist())

    trues = np.array(trues)
    preds = np.array(preds)
    probs = np.array(probs)

    dataset_size = max(1, len(trues))
    loss = running_loss / dataset_size if criterion is not None else None
    acc = float(np.mean(preds == trues))
    auc = float(roc_auc_score(trues, probs)) if len(np.unique(trues)) > 1 else 0.0
    ap = float(average_precision_score(trues, probs)) if len(np.unique(trues)) > 1 else 0.0

    out = {
        'loss': loss,
        'accuracy': acc,
        'auc': auc,
        'pr_auc': ap,
        'preds': preds,
        'trues': trues,
        'probs': probs
    }
    return out if return_probs else {k: v for k, v in out.items() if k not in ('preds', 'trues', 'probs')}


# =========================
# Main
# =========================
def main():
    set_seed(Config.SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    base_path = Path(__file__).parent
    df = pd.read_csv(base_path / 'participants_with_labels.csv')

    # Construct risk label example (as per earlier)
    df['risk_label'] = (
        (df['dementia_history_parents'].astype(float).fillna(0) > 0) |
        (df.get('CVLT_7', pd.Series([13.5]*len(df))).astype(float).fillna(13.5) < 13.5)
    ).astype(int)

    # Example feature dims (replace with real features as available)
    X_eeg = np.random.randn(len(df), 16).astype(np.float32)
    X_fmri = np.random.randn(len(df), 32).astype(np.float32)
    text_df = df[Config.TEXT_FEATURES].copy()
    y = df['risk_label'].values.astype(np.int64)

    # Split
    idx = np.arange(len(y))
    tr_val_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=Config.SEED, stratify=y)
    y_tr_val, y_te = y[tr_val_idx], y[te_idx]

    tr_idx_rel, va_idx_rel = train_test_split(
        np.arange(len(y_tr_val)), test_size=0.2, random_state=Config.SEED, stratify=y_tr_val
    )

    def sel(a, base_idx, rel_idx):
        return a[base_idx][rel_idx]

    X_tr_eeg = sel(X_eeg, tr_val_idx, tr_idx_rel)
    X_va_eeg = sel(X_eeg, tr_val_idx, va_idx_rel)
    X_te_eeg = X_eeg[te_idx]

    X_tr_fmri = sel(X_fmri, tr_val_idx, tr_idx_rel)
    X_va_fmri = sel(X_fmri, tr_val_idx, va_idx_rel)
    X_te_fmri = X_fmri[te_idx]

    text_tr = text_df.iloc[tr_val_idx].iloc[tr_idx_rel]
    text_va = text_df.iloc[tr_val_idx].iloc[va_idx_rel]
    text_te = text_df.iloc[te_idx]

    y_tr = y_tr_val[tr_idx_rel]
    y_va = y_tr_val[va_idx_rel]

    # Datasets
    train_ds = BrainDataset(X_tr_eeg, X_tr_fmri, text_tr, y_tr, augment=True, num_augmentations=Config.NUM_AUGMENTATIONS)
    val_ds = BrainDataset(X_va_eeg, X_va_fmri, text_va, y_va, augment=False)
    test_ds = BrainDataset(X_te_eeg, X_te_fmri, text_te, y_te, augment=False)

    # Sampler for imbalance
    sampler = make_weighted_sampler(y_tr)

    # Loaders
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, sampler=sampler,
        num_workers=2, pin_memory=(Config.DEVICE.type == 'cuda'), persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.VAL_BATCH_SIZE, shuffle=False,
        num_workers=1, pin_memory=(Config.DEVICE.type == 'cuda'), persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.TEST_BATCH_SIZE, shuffle=False,
        num_workers=1, pin_memory=(Config.DEVICE.type == 'cuda'), persistent_workers=True
    )

    # Model
    model = MultiModalNet(eeg_dim=X_tr_eeg.shape[1], fmri_dim=X_tr_fmri.shape[1]).to(Config.DEVICE)

    # Discriminative LR parameter groups with selective unfreezing
    param_groups = build_param_groups_discriminative(
        model, base_lr=Config.LEARNING_RATE_BASE, max_lr=Config.LEARNING_RATE_MAX
    )
    optimizer = optim.AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)

    # Loss
    class_counts = np.bincount(y_tr)
    pos_weight = class_counts[0] / max(1, class_counts[1]) if len(class_counts) > 1 else 1.0
    alpha_vec = torch.tensor([1.0, float(Config.ALPHA_POS if Config.ALPHA_POS is not None else pos_weight)],
                             device=Config.DEVICE)
    if Config.USE_FOCAL:
        criterion = FocalLoss(alpha=alpha_vec, gamma=Config.FOCAL_GAMMA, reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(weight=alpha_vec)

    # AMP
    scaler = torch.cuda.amp.GradScaler() if Config.DEVICE.type == 'cuda' else None

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Train loop with early stopping on PR-AUC
    best_val_prauc = -1
    best_state = None
    patience = 0

    for epoch in range(Config.EPOCHS):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scaler=scaler)
        val_metrics_full = evaluate(model, val_loader, criterion=None, return_probs=True)
        scheduler.step()

        epoch_time = time.time() - t0
        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics_full['accuracy']:.4f} val_auc={val_metrics_full['auc']:.4f} "
            f"val_pr_auc={val_metrics_full['pr_auc']:.4f} time={epoch_time:.2f}s"
        )

        if val_metrics_full['pr_auc'] > best_val_prauc:
            best_val_prauc = val_metrics_full['pr_auc']
            best_state = model.state_dict()
            patience = 0
            torch.save(best_state, "best_model_fused.pth")
        else:
            patience += 1

        if patience >= Config.PATIENCE:
            logger.info("Early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Threshold tuning on validation probabilities via ROC-G-Mean
    val_eval = evaluate(model, val_loader, criterion=None, return_probs=True)
    thr, gmean = tune_threshold_roc_gmean(val_eval['trues'], val_eval['probs'])
    logger.info(f"Chosen threshold={thr:.4f} with G-Mean={gmean:.4f} on validation")

    # Optionally calibrate probabilities (Platt scaling) using CV (meta step outside GPU)
    # This example uses logits via the model and fits a logistic regression calibrator
    # Note: for a neural net, one can export probabilities and re-calibrate off-line
    calibrator = None
    if Config.USE_CALIBRATION:
        # Collect validation probabilities as features; apply simple logistic calibration
        # Using scikit-learn CalibratedClassifierCV typically wraps a base estimator,
        # here we emulate by fitting a LR on probs to map to calibrated probs.
        lr_cal = LogisticRegression(max_iter=1000)
        y_val = val_eval['trues']
        p_val = val_eval['probs'].reshape(-1, 1)
        calibrator = lr_cal.fit(p_val, y_val)

    # Final evaluation on test set with tuned threshold and optional calibration
    test_eval = evaluate(model, test_loader, criterion=None, return_probs=True)
    p_test = test_eval['probs']
    if calibrator is not None:
        p_test = calibrator.predict_proba(p_test.reshape(-1, 1))[:, 1]
    y_test = test_eval['trues']
    y_pred_thresh = (p_test >= thr).astype(np.int64)

    final_metrics = {
        'loss': None,
        'accuracy': float(np.mean(y_pred_thresh == y_test)),
        'auc': float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else 0.0,
        'pr_auc': float(average_precision_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else 0.0,
        'classification_report': classification_report(y_test, y_pred_thresh),
        'confusion_matrix': confusion_matrix(y_test, y_pred_thresh).tolist(),
        'threshold': float(thr),
        'calibrated': bool(calibrator is not None)
    }
    logger.info(f"Final test metrics (thresholded): {json.dumps(final_metrics, indent=2)}")

    # Save results
    results_dir = base_path / 'results'
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'test_metrics_thresholded.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("Training and evaluation completed successfully")


if __name__ == "__main__":
    main()
