"""
Improved Multi-Modal Model with Enhanced Architecture and Training
This version includes:
- Better feature fusion strategies
- Advanced attention mechanisms
- Improved regularization
- Better data augmentation
- Learning rate scheduling
- Class balancing strategies
"""

from __future__ import annotations

import os
import logging
import json
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve
)
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedConfig:
    SEED = 42
    DROPOUT = 0.4  # Increased for better regularization
    WEIGHT_DECAY = 1e-2
    PATIENCE = 15  # Increased patience
    NUM_AUGMENTATIONS = 10  # More augmentations
    BATCH_SIZE = 32  # Larger batch size for GPU
    VAL_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    EPOCHS = 100  # More epochs with early stopping
    LEARNING_RATE = 5e-4  # Lower initial learning rate
    HIDDEN_DIM = 256  # Larger hidden dimensions
    NUM_HEADS = 8  # More attention heads
    NUM_LAYERS = 3  # Multi-layer encoders
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transformer configurations
    TRANSFORMER_MODEL = "distilbert-base-uncased"
    MAX_TEXT_LENGTH = 512
    TEXT_FEATURES = [
        'dementia_history_parents',
        'learning_deficits',
        'other_diseases',
        'drugs',
        'allergies'
    ]
    TEXT_HIDDEN_DIM = 768
    FUSION_HIDDEN_DIM = 512
    
    # Advanced training options
    USE_MIXUP = True  # Mixup augmentation
    MIXUP_ALPHA = 0.2
    USE_LABEL_SMOOTHING = True
    LABEL_SMOOTHING = 0.1
    USE_FOCAL_LOSS = True  # Better for imbalanced data
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    USE_COSINE_SCHEDULE = True
    
    # Class weighting
    USE_CLASS_WEIGHTS = True
    CLASS_WEIGHTS = [1.0, 2.5]  # Higher weight for positive class

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with residual connections."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + self.dropout(attn_out))

class ImprovedEncoder(nn.Module):
    """Multi-layer encoder with residual connections and attention."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Stacked encoder layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers - 1)
        ])
        
        # Self-attention
        self.attention = MultiHeadSelfAttention(hidden_dim, 4, dropout)
    
    def forward(self, x):
        x = self.input_proj(x)
        
        # Residual connections through layers
        for layer in self.layers:
            x = x + layer(x)
        
        # Self-attention with batch dimension handling
        x_attended = self.attention(x.unsqueeze(1))
        return x_attended.squeeze(1)

class ImprovedMultiModalNet(nn.Module):
    """Improved multi-modal network with better fusion."""
    def __init__(self, eeg_dim: int, fmri_dim: int):
        super().__init__()
        
        # Improved encoders
        self.eeg_encoder = ImprovedEncoder(
            eeg_dim, ImprovedConfig.HIDDEN_DIM, 
            ImprovedConfig.NUM_LAYERS, ImprovedConfig.DROPOUT
        )
        self.fmri_encoder = ImprovedEncoder(
            fmri_dim, ImprovedConfig.HIDDEN_DIM,
            ImprovedConfig.NUM_LAYERS, ImprovedConfig.DROPOUT
        )
        
        # Text encoder
        try:
            self.text_encoder = AutoModel.from_pretrained(ImprovedConfig.TRANSFORMER_MODEL)
            self.text_projector = nn.Sequential(
                nn.Linear(ImprovedConfig.TEXT_HIDDEN_DIM, ImprovedConfig.HIDDEN_DIM),
                nn.LayerNorm(ImprovedConfig.HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(ImprovedConfig.DROPOUT)
            )
            logger.info("Successfully loaded transformer model")
        except Exception as e:
            logger.warning(f"Failed to load transformer: {e}")
            raise
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            ImprovedConfig.HIDDEN_DIM,
            ImprovedConfig.NUM_HEADS,
            dropout=ImprovedConfig.DROPOUT,
            batch_first=True
        )
        
        # Improved fusion with gating mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(ImprovedConfig.HIDDEN_DIM * 3, 3),
            nn.Softmax(dim=1)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(ImprovedConfig.HIDDEN_DIM * 3, ImprovedConfig.FUSION_HIDDEN_DIM),
            nn.LayerNorm(ImprovedConfig.FUSION_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(ImprovedConfig.DROPOUT),
            nn.Linear(ImprovedConfig.FUSION_HIDDEN_DIM, ImprovedConfig.FUSION_HIDDEN_DIM // 2),
            nn.LayerNorm(ImprovedConfig.FUSION_HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(ImprovedConfig.DROPOUT)
        )
        
        # Classifier
        self.classifier = nn.Linear(ImprovedConfig.FUSION_HIDDEN_DIM // 2, 2)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, eeg, fmri, text_inputs):
        # Encode modalities
        eeg_features = self.eeg_encoder(eeg)
        fmri_features = self.fmri_encoder(fmri)
        
        text_outputs = self.text_encoder(**text_inputs)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projector(text_features)
        
        # Stack features for cross-attention
        features_stack = torch.stack([eeg_features, fmri_features, text_features], dim=1)
        
        # Cross-modal attention
        attended_features, _ = self.cross_attention(features_stack, features_stack, features_stack)
        
        # Flatten attended features
        attended_flat = attended_features.reshape(attended_features.size(0), -1)
        
        # Gated fusion
        gates = self.fusion_gate(attended_flat)
        gates = gates.unsqueeze(-1)
        
        weighted_features = attended_features * gates
        weighted_flat = weighted_features.reshape(weighted_features.size(0), -1)
        
        # Fusion and classification
        fused = self.fusion_layer(weighted_flat)
        return self.classifier(fused)

def mixup_data(x_eeg, x_fmri, text_inputs, y, alpha=0.2):
    """Mixup augmentation for better generalization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x_eeg.size(0)
    index = torch.randperm(batch_size).to(x_eeg.device)
    
    mixed_eeg = lam * x_eeg + (1 - lam) * x_eeg[index]
    mixed_fmri = lam * x_fmri + (1 - lam) * x_fmri[index]
    
    # Can't mixup text directly, use original
    y_a, y_b = y, y[index]
    
    return mixed_eeg, mixed_fmri, text_inputs, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss computation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_epoch_improved(model, data_loader, optimizer, criterion, scaler=None, epoch=0):
    """Improved training loop with mixup and better logging."""
    model.train()
    running_loss = 0.0
    predictions = []
    true_labels = []
    
    use_amp = (ImprovedConfig.DEVICE.type == 'cuda')
    use_mixup = ImprovedConfig.USE_MIXUP and (epoch > 5)  # Start mixup after initial epochs
    
    pbar = tqdm(data_loader, desc=f"Train Epoch {epoch+1}", leave=False)
    for batch_idx, (eeg, fmri, text_inputs, labels) in enumerate(pbar):
        eeg = eeg.to(ImprovedConfig.DEVICE, non_blocking=True)
        fmri = fmri.to(ImprovedConfig.DEVICE, non_blocking=True)
        labels = labels.to(ImprovedConfig.DEVICE, non_blocking=True)
        
        text_inputs = {
            'input_ids': text_inputs['input_ids'].to(ImprovedConfig.DEVICE, non_blocking=True),
            'attention_mask': text_inputs['attention_mask'].to(ImprovedConfig.DEVICE, non_blocking=True)
        }
        
        optimizer.zero_grad()
        
        # Apply mixup
        if use_mixup and np.random.rand() < 0.5:
            eeg, fmri, text_inputs, labels_a, labels_b, lam = mixup_data(
                eeg, fmri, text_inputs, labels, ImprovedConfig.MIXUP_ALPHA
            )
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(eeg, fmri, text_inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(eeg, fmri, text_inputs)
                loss = criterion(outputs, labels)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return {
        'loss': running_loss / len(data_loader.dataset),
        'accuracy': np.mean(np.array(predictions) == np.array(true_labels)),
        'predictions': predictions,
        'true_labels': true_labels
    }

def evaluate_model_improved(model, data_loader):
    """Improved evaluation with probability outputs."""
    model.eval()
    predictions = []
    probabilities = []
    true_labels = []
    running_loss = 0.0
    
    if ImprovedConfig.USE_FOCAL_LOSS:
        criterion = FocalLoss(
            alpha=ImprovedConfig.FOCAL_ALPHA,
            gamma=ImprovedConfig.FOCAL_GAMMA
        )
    else:
        weights = torch.FloatTensor(ImprovedConfig.CLASS_WEIGHTS).to(ImprovedConfig.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
    
    with torch.no_grad():
        for eeg, fmri, text_inputs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            eeg = eeg.to(ImprovedConfig.DEVICE)
            fmri = fmri.to(ImprovedConfig.DEVICE)
            labels = labels.to(ImprovedConfig.DEVICE)
            
            text_inputs = {
                'input_ids': text_inputs['input_ids'].to(ImprovedConfig.DEVICE),
                'attention_mask': text_inputs['attention_mask'].to(ImprovedConfig.DEVICE)
            }
            
            outputs = model(eeg, fmri, text_inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            
            probs = F.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
            _, preds = torch.max(outputs, 1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    true_labels = np.array(true_labels)
    
    return {
        'loss': running_loss / len(data_loader.dataset),
        'accuracy': np.mean(predictions == true_labels),
        'auc': roc_auc_score(true_labels, probabilities) if len(np.unique(true_labels)) > 1 else 0.0,
        'avg_precision': average_precision_score(true_labels, probabilities) if len(np.unique(true_labels)) > 1 else 0.0,
        'predictions': predictions,
        'probabilities': probabilities,
        'true_labels': true_labels,
        'classification_report': classification_report(true_labels, predictions),
        'confusion_matrix': confusion_matrix(true_labels, predictions)
    }

def main_improved():
    try:
        # Set random seeds
        torch.manual_seed(ImprovedConfig.SEED)
        np.random.seed(ImprovedConfig.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ImprovedConfig.SEED)
        
        logger.info(f"Using device: {ImprovedConfig.DEVICE}")
        
        # Load data (same as before)
        base_path = Path(__file__).parent
        participants_df = pd.read_csv(base_path / 'participants_with_labels.csv')
        logger.info(f"Loaded data for {len(participants_df)} participants")
        
        # Create labels
        participants_df['risk_label'] = (
            (participants_df['dementia_history_parents'].astype(float).fillna(0) > 0) | 
            (participants_df.get('CVLT_7', pd.Series([13.5]*len(participants_df))).astype(float).fillna(13.5) < 13.5)
        ).astype(int)
        
        label_counts = participants_df['risk_label'].value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")
        
        # Prepare data (replace with real features when available)
        X_eeg = np.random.randn(len(participants_df), 16).astype(np.float32)
        X_fmri = np.random.randn(len(participants_df), 32).astype(np.float32)
        text_data = participants_df[ImprovedConfig.TEXT_FEATURES]
        y = participants_df['risk_label'].values
        
        # Split data
        indices = np.arange(len(y))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=np.asarray(y), random_state=ImprovedConfig.SEED
        )
        
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.2, stratify=np.asarray(y)[train_val_idx], 
            random_state=ImprovedConfig.SEED
        )
        
        # Create datasets (reuse BrainDataset from original code)
        from hybrid_model import BrainDataset
        
        train_dataset = BrainDataset(
            X_eeg[train_idx], X_fmri[train_idx], text_data.iloc[train_idx], y[train_idx],
            augment=True, num_augmentations=ImprovedConfig.NUM_AUGMENTATIONS
        )
        
        val_dataset = BrainDataset(
            X_eeg[val_idx], X_fmri[val_idx], text_data.iloc[val_idx], y[val_idx],
            augment=False
        )
        
        test_dataset = BrainDataset(
            X_eeg[test_idx], X_fmri[test_idx], text_data.iloc[test_idx], y[test_idx],
            augment=False
        )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=ImprovedConfig.BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=(ImprovedConfig.DEVICE.type == 'cuda'),
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=ImprovedConfig.VAL_BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=(ImprovedConfig.DEVICE.type == 'cuda'),
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=ImprovedConfig.TEST_BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=(ImprovedConfig.DEVICE.type == 'cuda'),
            persistent_workers=True
        )
        
        # Model
        model = ImprovedMultiModalNet(
            eeg_dim=X_eeg.shape[1],
            fmri_dim=X_fmri.shape[1]
        ).to(ImprovedConfig.DEVICE)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss function
        if ImprovedConfig.USE_FOCAL_LOSS:
            criterion = FocalLoss(
                alpha=ImprovedConfig.FOCAL_ALPHA,
                gamma=ImprovedConfig.FOCAL_GAMMA
            )
            logger.info("Using Focal Loss")
        else:
            weights = torch.FloatTensor(ImprovedConfig.CLASS_WEIGHTS).to(ImprovedConfig.DEVICE)
            criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=ImprovedConfig.LABEL_SMOOTHING if ImprovedConfig.USE_LABEL_SMOOTHING else 0.0)
            logger.info("Using Cross-Entropy Loss with class weights")
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=ImprovedConfig.LEARNING_RATE,
            weight_decay=ImprovedConfig.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        if ImprovedConfig.USE_COSINE_SCHEDULE:
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            logger.info("Using Cosine Annealing scheduler")
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
            logger.info("Using ReduceLROnPlateau scheduler")
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if ImprovedConfig.DEVICE.type == 'cuda' else None
        
        # Training loop
        best_val_auc = 0
        best_model_state = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_acc': []}
        
        epoch_bar = tqdm(range(ImprovedConfig.EPOCHS), desc="Epochs")
        for epoch in epoch_bar:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            t0 = time.perf_counter()
            
            # Train
            train_metrics = train_epoch_improved(model, train_loader, optimizer, criterion, scaler, epoch)
            
            # Validate
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            val_metrics = evaluate_model_improved(model, val_loader)
            
            epoch_time = time.perf_counter() - t0
            
            # Update scheduler: call with metric only for ReduceLROnPlateau, otherwise call without arguments
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['auc'])  # Pass validation AUC as metric for ReduceLROnPlateau
            else:
                scheduler.step()  # CosineAnnealingWarmRestarts and similar schedulers do not expect a metric
            
            # Track history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_auc'].append(val_metrics['auc'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Save best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, 'best_model_improved.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Logging
            gpu_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            
            epoch_bar.set_postfix({
                'train_loss': f"{train_metrics['loss']:.4f}",
                'val_loss': f"{val_metrics['loss']:.4f}",
                'val_auc': f"{val_metrics['auc']:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
                'gpu_mb': f"{gpu_mem:.1f}"
            })
            
            logger.info(
                f"Epoch {epoch+1}/{ImprovedConfig.EPOCHS}: "
                f"Time={epoch_time:.2f}s, "
                f"Train Loss={train_metrics['loss']:.4f}, "
                f"Val Loss={val_metrics['loss']:.4f}, "
                f"Val Acc={val_metrics['accuracy']:.4f}, "
                f"Val AUC={val_metrics['auc']:.4f}, "
                f"Val AP={val_metrics['avg_precision']:.4f}, "
                f"LR={optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Early stopping
            if patience_counter >= ImprovedConfig.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model and evaluate on test set
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        test_metrics = evaluate_model_improved(model, test_loader)
        
        logger.info("\n" + "="*60)
        logger.info("FINAL TEST RESULTS")
        logger.info("="*60)
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
        logger.info(f"Test Average Precision: {test_metrics['avg_precision']:.4f}")
        logger.info(f"\n{test_metrics['classification_report']}")
        logger.info(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")
        
        # Save results
        results_dir = base_path / 'results_improved'
        results_dir.mkdir(exist_ok=True)
        
        results = {
            'test_metrics': {
                'accuracy': float(test_metrics['accuracy']),
                'auc': float(test_metrics['auc']),
                'avg_precision': float(test_metrics['avg_precision']),
                'classification_report': test_metrics['classification_report'],
                'confusion_matrix': test_metrics['confusion_matrix'].tolist()
            },
            'best_val_auc': float(best_val_auc),
            'training_history': history
        }
        
        with open(results_dir / 'test_metrics.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save predictions with probabilities
        test_predictions_df = pd.DataFrame({
            'participant_id': participants_df.iloc[test_idx]['participant_id'].values if 'participant_id' in participants_df.columns else test_idx,
            'y_true': test_metrics['true_labels'],
            'y_pred': test_metrics['predictions'],
            'y_prob': test_metrics['probabilities']
        })
        test_predictions_df.to_csv(results_dir / 'test_predictions.csv', index=False)
        
        logger.info(f"\nResults saved to {results_dir}")
        logger.info("Improved training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in improved training: {str(e)}")
        raise

if __name__ == "__main__":
    main_improved()
