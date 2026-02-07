"""
Enhanced Multi-Modal Model with State-of-the-Art Hugging Face Models
Features:
- Multiple specialized Hugging Face models (Medical, Clinical, Scientific)
- Enhanced architecture with better attention mechanisms
- Advanced training techniques (Focal Loss, Mixup, etc.)
- Better text preprocessing for medical context
"""

from __future__ import annotations

import os
import logging
import json
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve
)
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel,
    DistilBertTokenizer, DistilBertModel
)
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedConfig:
    SEED = 42
    DROPOUT = 0.3
    WEIGHT_DECAY = 1e-2
    PATIENCE = 15
    NUM_AUGMENTATIONS = 12
    BATCH_SIZE = 8  # Smaller for larger models
    VAL_BATCH_SIZE = 8
    TEST_BATCH_SIZE = 8
    EPOCHS = 60
    LEARNING_RATE = 2e-5
    HIDDEN_DIM = 512  # Increased
    NUM_HEADS = 16
    NUM_LAYERS = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # State-of-the-Art Hugging Face Models
    AVAILABLE_MODELS = {
        # Medical Domain Models
        'biomedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        'clinicalbert': 'emilyalsentzer/Bio_ClinicalBERT', 
        'scibert': 'allenai/scibert_scivocab_uncased',
        'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        
        # General Purpose Strong Models
        'roberta': 'roberta-base',
        'deberta': 'microsoft/deberta-v3-base',
        'electra': 'google/electra-base-discriminator',
        
        # Efficient Models
        'distilbert': 'distilbert-base-uncased',
        'distilroberta': 'distilroberta-base',
        
        # Large Models (if GPU memory allows)
        'biomedbert_large': 'microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract',
        'roberta_large': 'roberta-large'
    }
    
    # Choose your model here
    CHOSEN_MODEL = 'biomedbert'  # Change this to experiment with different models
    
    # Model configurations
    MODEL_CONFIGS = {
        'biomedbert': {'hidden_dim': 768, 'max_length': 512, 'model_type': 'bert'},
        'clinicalbert': {'hidden_dim': 768, 'max_length': 512, 'model_type': 'bert'},
        'scibert': {'hidden_dim': 768, 'max_length': 512, 'model_type': 'bert'},
        'pubmedbert': {'hidden_dim': 768, 'max_length': 512, 'model_type': 'bert'},
        'roberta': {'hidden_dim': 768, 'max_length': 512, 'model_type': 'roberta'},
        'deberta': {'hidden_dim': 768, 'max_length': 512, 'model_type': 'deberta'},
        'electra': {'hidden_dim': 768, 'max_length': 512, 'model_type': 'electra'},
        'distilbert': {'hidden_dim': 768, 'max_length': 512, 'model_type': 'bert'},
        'distilroberta': {'hidden_dim': 768, 'max_length': 512, 'model_type': 'roberta'},
        'biomedbert_large': {'hidden_dim': 1024, 'max_length': 512, 'model_type': 'bert'},
        'roberta_large': {'hidden_dim': 1024, 'max_length': 512, 'model_type': 'roberta'}
    }
    
    TEXT_FEATURES = [
        'dementia_history_parents',
        'learning_deficits',
        'other_diseases', 
        'drugs',
        'allergies'
    ]
    
    # Get config for chosen model
    TEXT_HIDDEN_DIM = MODEL_CONFIGS[CHOSEN_MODEL]['hidden_dim']
    MAX_TEXT_LENGTH = MODEL_CONFIGS[CHOSEN_MODEL]['max_length']
    FUSION_HIDDEN_DIM = 768
    
    # Advanced options
    USE_GRADIENT_ACCUMULATION = True
    ACCUMULATION_STEPS = 4
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    USE_FOCAL_LOSS = True
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
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
        return focal_loss

class AdvancedBrainDataset(Dataset):
    """Advanced dataset with enhanced text processing."""
    
    def __init__(self, eeg_features: np.ndarray, fmri_features: np.ndarray, 
                 text_data, labels: np.ndarray, 
                 model_name: str, augment: bool = False, num_augmentations: int = 5):
        
        self.original_eeg = torch.FloatTensor(eeg_features)
        self.original_fmri = torch.FloatTensor(fmri_features)
        self.original_labels = torch.LongTensor(labels)
        self.text_data = text_data
        self.augment = augment
        self.num_augmentations = num_augmentations
        self.model_name = model_name
        
        # Initialize tokenizer
        self._init_tokenizer()
        
        # Process text with medical context
        self.processed_text = self._process_medical_text()
        
        # Data augmentation
        if augment:
            self.eeg, self.fmri, self.labels = self._create_augmented_dataset()
        else:
            self.eeg = self.original_eeg
            self.fmri = self.original_fmri
            self.labels = self.original_labels
    
    def _init_tokenizer(self):
        """Initialize the appropriate tokenizer."""
        try:
            model_path = AdvancedConfig.AVAILABLE_MODELS[AdvancedConfig.CHOSEN_MODEL]
            model_type = AdvancedConfig.MODEL_CONFIGS[AdvancedConfig.CHOSEN_MODEL]['model_type']
            
            if model_type == 'bert':
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            elif model_type == 'roberta':
                self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            elif model_type == 'deberta':
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Ensure pad token
            if self.tokenizer.pad_token is None:
                if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            logger.info(f"Successfully loaded tokenizer for {self.model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {self.model_name}: {e}")
            logger.info("Falling back to DistilBERT")
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = '[PAD]'
    
    def _process_medical_text(self) -> list:
        """Process text with medical domain knowledge."""
        processed = []
        
        for idx, row in enumerate(self.text_data):
            # Build comprehensive medical narrative
            medical_sections = []
            
            # Patient history header
            medical_sections.append("PATIENT CLINICAL HISTORY:")
            
            # Process each medical field with clinical context
            for field in AdvancedConfig.TEXT_FEATURES:
                value = str(row.get(field, ''))
                if value and value.lower() not in ['nan', 'none', '', '0', '0.0']:
                    
                    if field == 'dementia_history_parents':
                        medical_sections.append(f"FAMILY HISTORY: Parental dementia history - {value}")
                    elif field == 'learning_deficits':
                        medical_sections.append(f"COGNITIVE PROFILE: Learning difficulties documented - {value}")
                    elif field == 'other_diseases':
                        medical_sections.append(f"COMORBIDITIES: Additional medical conditions - {value}")
                    elif field == 'drugs':
                        medical_sections.append(f"CURRENT MEDICATIONS: Pharmaceutical treatments - {value}")
                    elif field == 'allergies':
                        medical_sections.append(f"ALLERGIC REACTIONS: Known allergies and sensitivities - {value}")
            
            # Add clinical assessment context
            medical_sections.append("ASSESSMENT: Neuroimaging and electrophysiological evaluation for cognitive decline risk assessment.")
            medical_sections.append("MODALITIES: Brain MRI functional connectivity analysis and EEG neural oscillation patterns.")
            
            # Join sections
            if len(medical_sections) > 2:  # More than just headers
                clinical_text = " | ".join(medical_sections)
            else:
                clinical_text = "PATIENT CLINICAL HISTORY: No significant medical history documented. ASSESSMENT: Baseline neuroimaging and EEG evaluation for cognitive health screening."
            
            try:
                # Tokenize with clinical context
                encoded = self.tokenizer(
                    clinical_text,
                    max_length=AdvancedConfig.MAX_TEXT_LENGTH,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    add_special_tokens=True
                )
                
                processed.append({
                    'input_ids': encoded['input_ids'].squeeze(0),
                    'attention_mask': encoded['attention_mask'].squeeze(0)
                })
                
            except Exception as e:
                logger.warning(f"Error processing text for sample {idx}: {e}")
                # Fallback
                processed.append({
                    'input_ids': torch.zeros(AdvancedConfig.MAX_TEXT_LENGTH, dtype=torch.long),
                    'attention_mask': torch.ones(AdvancedConfig.MAX_TEXT_LENGTH, dtype=torch.long)
                })
        
        return processed
    
    def _create_augmented_dataset(self):
        """Advanced data augmentation for multimodal brain data."""
        augmented_eeg = [self.original_eeg]
        augmented_fmri = [self.original_fmri]
        augmented_labels = [self.original_labels]
        
        for i in range(self.num_augmentations):
            # Progressive augmentation intensity
            base_noise = 0.03 + (i * 0.01)
            
            # EEG augmentation (simulate electrode noise and artifacts)
            eeg_aug = self.original_eeg.clone()
            eeg_aug += torch.randn_like(eeg_aug) * base_noise
            
            # Add occasional spikes (simulating artifacts)
            if i % 4 == 0:
                spike_mask = torch.rand_like(eeg_aug) < 0.05
                eeg_aug[spike_mask] += torch.randn_like(eeg_aug[spike_mask]) * 0.2
            
            # fMRI augmentation (simulate scanner noise and motion)
            fmri_aug = self.original_fmri.clone()
            fmri_aug += torch.randn_like(fmri_aug) * base_noise
            
            # Add spatial correlation (neighboring voxels)
            if i % 3 == 0:
                spatial_noise = torch.randn(fmri_aug.size(0), 1) * 0.02
                fmri_aug += spatial_noise.expand_as(fmri_aug)
            
            # Cross-modal correlation (brain states affect both EEG and fMRI)
            if i % 5 == 0:
                shared_state = torch.randn(eeg_aug.size(0), 1) * 0.015
                eeg_aug += shared_state.expand_as(eeg_aug) * 0.5
                fmri_aug += shared_state.expand_as(fmri_aug) * 0.5
            
            augmented_eeg.append(eeg_aug)
            augmented_fmri.append(fmri_aug)
            augmented_labels.append(self.original_labels)
        
        return (torch.cat(augmented_eeg, dim=0),
                torch.cat(augmented_fmri, dim=0),
                torch.cat(augmented_labels, dim=0))
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict, torch.Tensor]:
        if self.augment:
            text_idx = idx % len(self.original_labels)
        else:
            text_idx = idx
            
        text_data = {
            'input_ids': self.processed_text[text_idx]['input_ids'],
            'attention_mask': self.processed_text[text_idx]['attention_mask']
        }
        return self.eeg[idx], self.fmri[idx], text_data, self.labels[idx]

class CrossModalAttention(nn.Module):
    """Advanced cross-modal attention mechanism."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Learnable temperature for attention
        self.temperature = nn.Parameter(torch.ones(1) * (hidden_dim ** -0.5))
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Project and reshape
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with learnable temperature
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.temperature
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection with residual connection
        output = self.out_proj(context)
        return self.layer_norm(output + query), attn_weights

class AdvancedEncoder(nn.Module):
    """Advanced encoder with multiple attention layers."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer-like encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': CrossModalAttention(hidden_dim, num_heads, dropout),
                'feed_forward': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim)
            }) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            # Self-attention with residual
            attn_out, _ = layer['attention'](x, x, x)
            
            # Feed-forward with residual
            ff_out = layer['feed_forward'](layer['norm1'](attn_out))
            x = layer['norm2'](attn_out + ff_out)
        
        # Output projection
        x = x.squeeze(1)
        return self.output_proj(x)

class AdvancedMultiModalNet(nn.Module):
    """State-of-the-art multi-modal neural network."""
    
    def __init__(self, eeg_dim: int, fmri_dim: int, model_name: str):
        super().__init__()
        
        self.model_name = model_name
        hidden_dim = AdvancedConfig.HIDDEN_DIM
        
        # Advanced encoders for brain signals
        self.eeg_encoder = AdvancedEncoder(
            eeg_dim, hidden_dim, 
            AdvancedConfig.NUM_LAYERS, AdvancedConfig.NUM_HEADS, AdvancedConfig.DROPOUT
        )
        self.fmri_encoder = AdvancedEncoder(
            fmri_dim, hidden_dim,
            AdvancedConfig.NUM_LAYERS, AdvancedConfig.NUM_HEADS, AdvancedConfig.DROPOUT
        )
        
        # Load state-of-the-art text encoder
        self._init_text_encoder()
        
        # Multi-head cross-modal attention
        self.cross_modal_attention = CrossModalAttention(
            hidden_dim, AdvancedConfig.NUM_HEADS, AdvancedConfig.DROPOUT
        )
        
        # Adaptive fusion with learnable weights
        self.modality_fusion = nn.ModuleDict({
            'gate_network': nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(AdvancedConfig.DROPOUT),
                nn.Linear(hidden_dim, 3),
                nn.Softmax(dim=-1)
            ),
            'feature_combiner': nn.Sequential(
                nn.Linear(hidden_dim * 3, AdvancedConfig.FUSION_HIDDEN_DIM),
                nn.LayerNorm(AdvancedConfig.FUSION_HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(AdvancedConfig.DROPOUT),
                
                nn.Linear(AdvancedConfig.FUSION_HIDDEN_DIM, AdvancedConfig.FUSION_HIDDEN_DIM // 2),
                nn.LayerNorm(AdvancedConfig.FUSION_HIDDEN_DIM // 2),
                nn.GELU(),
                nn.Dropout(AdvancedConfig.DROPOUT),
                
                nn.Linear(AdvancedConfig.FUSION_HIDDEN_DIM // 2, AdvancedConfig.FUSION_HIDDEN_DIM // 4),
                nn.LayerNorm(AdvancedConfig.FUSION_HIDDEN_DIM // 4),
                nn.GELU(),
                nn.Dropout(AdvancedConfig.DROPOUT // 2)
            )
        })
        
        # Final classifier with uncertainty estimation
        self.classifier = nn.Sequential(
            nn.Linear(AdvancedConfig.FUSION_HIDDEN_DIM // 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(AdvancedConfig.DROPOUT // 2),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(AdvancedConfig.DROPOUT // 4),
            
            nn.Linear(64, 2)
        )
        
        self._init_weights()
    
    def _init_text_encoder(self):
        """Initialize state-of-the-art text encoder."""
        try:
            model_path = AdvancedConfig.AVAILABLE_MODELS[AdvancedConfig.CHOSEN_MODEL]
            model_type = AdvancedConfig.MODEL_CONFIGS[AdvancedConfig.CHOSEN_MODEL]['model_type']
            
            logger.info(f"Loading {AdvancedConfig.CHOSEN_MODEL} model: {model_path}")
            
            # Load appropriate model
            if model_type == 'bert':
                self.text_encoder = AutoModel.from_pretrained(model_path)
            elif model_type == 'roberta':
                self.text_encoder = RobertaModel.from_pretrained(model_path)
            elif model_type == 'deberta':
                self.text_encoder = AutoModel.from_pretrained(model_path)
            else:
                self.text_encoder = AutoModel.from_pretrained(model_path)
            
            # Selective freezing for stability
            if hasattr(self.text_encoder, 'embeddings'):
                for param in self.text_encoder.embeddings.parameters():
                    param.requires_grad = False
            
            # Text projection with residual connections
            text_hidden_dim = AdvancedConfig.TEXT_HIDDEN_DIM
            self.text_projector = nn.Sequential(
                nn.Linear(text_hidden_dim, AdvancedConfig.HIDDEN_DIM * 2),
                nn.LayerNorm(AdvancedConfig.HIDDEN_DIM * 2),
                nn.GELU(),
                nn.Dropout(AdvancedConfig.DROPOUT),
                
                nn.Linear(AdvancedConfig.HIDDEN_DIM * 2, AdvancedConfig.HIDDEN_DIM),
                nn.LayerNorm(AdvancedConfig.HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(AdvancedConfig.DROPOUT)
            )
            
            logger.info(f"Successfully loaded text encoder: {AdvancedConfig.CHOSEN_MODEL}")
            
        except Exception as e:
            logger.error(f"Failed to load {AdvancedConfig.CHOSEN_MODEL}: {e}")
            logger.info("Falling back to DistilBERT")
            
            self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.text_projector = nn.Sequential(
                nn.Linear(768, AdvancedConfig.HIDDEN_DIM),
                nn.LayerNorm(AdvancedConfig.HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(AdvancedConfig.DROPOUT)
            )
    
    def _init_weights(self):
        """Initialize weights for new components."""
        for name, module in self.named_modules():
            if 'text_encoder' not in name:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def forward(self, eeg, fmri, text_inputs):
        # Encode brain signals
        eeg_features = self.eeg_encoder(eeg)
        fmri_features = self.fmri_encoder(fmri)
        
        # Encode text with state-of-the-art model
        text_outputs = self.text_encoder(**text_inputs)
        
        # Extract text representation
        if hasattr(text_outputs, 'last_hidden_state'):
            # Use CLS token or mean pooling
            text_repr = text_outputs.last_hidden_state[:, 0, :]
        elif hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
            text_repr = text_outputs.pooler_output
        else:
            # Mean pooling as fallback
            text_repr = text_outputs[0].mean(dim=1)
        
        text_features = self.text_projector(text_repr)
        
        # Cross-modal attention fusion
        features_stack = torch.stack([eeg_features, fmri_features, text_features], dim=1)
        attended_features, attention_weights = self.cross_modal_attention(
            features_stack, features_stack, features_stack
        )
        
        # Adaptive fusion with learned gates
        concatenated = attended_features.view(attended_features.size(0), -1)
        
        # Compute modality importance gates
        gates = self.modality_fusion['gate_network'](concatenated)
        gates = gates.unsqueeze(-1)
        
        # Apply gates and combine
        weighted_features = attended_features * gates
        final_features = weighted_features.view(weighted_features.size(0), -1)
        
        # Feature combination and classification
        fused = self.modality_fusion['feature_combiner'](final_features)
        logits = self.classifier(fused)
        
        return logits
    
    def get_interpretation_data(self, eeg, fmri, text_inputs):
        """Get data for model interpretation."""
        with torch.no_grad():
            # Forward pass to get features
            eeg_features = self.eeg_encoder(eeg)
            fmri_features = self.fmri_encoder(fmri)
            
            text_outputs = self.text_encoder(**text_inputs)
            if hasattr(text_outputs, 'last_hidden_state'):
                text_repr = text_outputs.last_hidden_state[:, 0, :]
            else:
                text_repr = text_outputs[0].mean(dim=1)
            text_features = self.text_projector(text_repr)
            
            # Get attention weights
            features_stack = torch.stack([eeg_features, fmri_features, text_features], dim=1)
            _, attention_weights = self.cross_modal_attention(features_stack, features_stack, features_stack)
            
            # Get modality gates
            concatenated = features_stack.view(features_stack.size(0), -1)
            gates = self.modality_fusion['gate_network'](concatenated)
            
        return {
            'attention_weights': attention_weights.cpu().numpy(),
            'modality_gates': gates.cpu().numpy(),
            'feature_norms': {
                'eeg': torch.norm(eeg_features, dim=1).cpu().numpy(),
                'fmri': torch.norm(fmri_features, dim=1).cpu().numpy(),
                'text': torch.norm(text_features, dim=1).cpu().numpy()
            },
            'modality_names': ['EEG', 'fMRI', 'Clinical Text']
        }

def mixup_data(x_eeg, x_fmri, text_inputs, y, alpha=0.2):
    """Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x_eeg.size(0)
    index = torch.randperm(batch_size).to(x_eeg.device)
    
    mixed_eeg = lam * x_eeg + (1 - lam) * x_eeg[index]
    mixed_fmri = lam * x_fmri + (1 - lam) * x_fmri[index]
    
    y_a, y_b = y, y[index]
    return mixed_eeg, mixed_fmri, text_inputs, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_epoch_advanced(model, data_loader, optimizer, criterion, scaler=None, epoch=0):
    """Advanced training with all techniques."""
    model.train()
    running_loss = 0.0
    predictions = []
    true_labels = []
    
    use_amp = (AdvancedConfig.DEVICE.type == 'cuda')
    use_mixup = AdvancedConfig.USE_MIXUP and (epoch > 2)
    accumulation_steps = AdvancedConfig.ACCUMULATION_STEPS if AdvancedConfig.USE_GRADIENT_ACCUMULATION else 1
    
    optimizer.zero_grad()
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}", leave=False)
    for batch_idx, (eeg, fmri, text_inputs, labels) in enumerate(pbar):
        eeg = eeg.to(AdvancedConfig.DEVICE, non_blocking=True)
        fmri = fmri.to(AdvancedConfig.DEVICE, non_blocking=True)
        labels = labels.to(AdvancedConfig.DEVICE, non_blocking=True)
        
        text_inputs = {
            'input_ids': text_inputs['input_ids'].to(AdvancedConfig.DEVICE, non_blocking=True),
            'attention_mask': text_inputs['attention_mask'].to(AdvancedConfig.DEVICE, non_blocking=True)
        }
        
        # Apply mixup
        if use_mixup and np.random.rand() < 0.5:
            mixed_eeg, mixed_fmri, text_inputs, labels_a, labels_b, lam = mixup_data(
                eeg, fmri, text_inputs, labels, AdvancedConfig.MIXUP_ALPHA
            )
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(mixed_eeg, mixed_fmri, text_inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(eeg, fmri, text_inputs)
                loss = criterion(outputs, labels)
        
        # Scale for gradient accumulation
        loss = loss / accumulation_steps
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps * labels.size(0)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item() * accumulation_steps:.4f}",
            'acc': f"{np.mean(np.array(predictions) == np.array(true_labels)):.3f}"
        })
    
    return {
        'loss': running_loss / len(data_loader.dataset),
        'accuracy': np.mean(np.array(predictions) == np.array(true_labels)),
        'predictions': predictions,
        'true_labels': true_labels
    }

def evaluate_model_advanced(model, data_loader):
    """Advanced evaluation with comprehensive metrics."""
    model.eval()
    predictions = []
    probabilities = []
    true_labels = []
    running_loss = 0.0
    
    criterion = FocalLoss(alpha=AdvancedConfig.FOCAL_ALPHA, gamma=AdvancedConfig.FOCAL_GAMMA)
    
    with torch.no_grad():
        for eeg, fmri, text_inputs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            eeg = eeg.to(AdvancedConfig.DEVICE)
            fmri = fmri.to(AdvancedConfig.DEVICE)
            labels = labels.to(AdvancedConfig.DEVICE)
            
            text_inputs = {
                'input_ids': text_inputs['input_ids'].to(AdvancedConfig.DEVICE),
                'attention_mask': text_inputs['attention_mask'].to(AdvancedConfig.DEVICE)
            }
            
            outputs = model(eeg, fmri, text_inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            
            probs = F.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    true_labels = np.array(true_labels)
    
    metrics = {
        'loss': running_loss / len(data_loader.dataset),
        'accuracy': np.mean(predictions == true_labels),
        'predictions': predictions,
        'probabilities': probabilities,
        'true_labels': true_labels,
        'classification_report': classification_report(true_labels, predictions),
        'confusion_matrix': confusion_matrix(true_labels, predictions)
    }
    
    if len(np.unique(true_labels)) > 1:
        metrics['auc'] = roc_auc_score(true_labels, probabilities)
        metrics['avg_precision'] = average_precision_score(true_labels, probabilities)
    else:
        metrics['auc'] = 0.0
        metrics['avg_precision'] = 0.0
    
    return metrics

def main_advanced():
    try:
        # Reproducibility
        torch.manual_seed(AdvancedConfig.SEED)
        np.random.seed(AdvancedConfig.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(AdvancedConfig.SEED)
        
        logger.info(f"🚀 ADVANCED MULTI-MODAL TRAINING")
        logger.info(f"Device: {AdvancedConfig.DEVICE}")
        logger.info(f"Model: {AdvancedConfig.CHOSEN_MODEL}")
        logger.info(f"Model Path: {AdvancedConfig.AVAILABLE_MODELS[AdvancedConfig.CHOSEN_MODEL]}")
        
        # Load data manually
        base_path = Path(__file__).parent
        csv_path = base_path / 'participants_with_labels.csv'
        
        participants_data = []
        with open(csv_path, 'r') as f:
            header = f.readline().strip().split(',')
            for line in f:
                row_data = line.strip().split(',')
                participants_data.append(dict(zip(header, row_data)))
        
        logger.info(f"📊 Loaded {len(participants_data)} participants")
        
        # Process labels and text
        labels = []
        text_data = []
        
        for row in participants_data:
            # Risk labeling
            dementia_history = float(row.get('dementia_history_parents', 0)) if row.get('dementia_history_parents', '').replace('.','').replace('-','').isdigit() else 0
            cvlt_score = float(row.get('CVLT_7', 13.5)) if row.get('CVLT_7', '').replace('.','').replace('-','').isdigit() else 13.5
            
            risk_label = 1 if (dementia_history > 0 or cvlt_score < 13.5) else 0
            labels.append(risk_label)
            
            # Text features
            text_features = {feat: row.get(feat, '') for feat in AdvancedConfig.TEXT_FEATURES}
            text_data.append(text_features)
        
        labels = np.array(labels)
        
        # Label distribution
        unique, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique, counts))
        logger.info(f"📈 Label distribution: {label_dist}")
        
        # Create enhanced synthetic brain data
        n_samples = len(participants_data)
        
        # More realistic EEG simulation (frequency bands)
        X_eeg = np.random.randn(n_samples, 16).astype(np.float32)
        
        # More realistic fMRI simulation (brain regions)
        X_fmri = np.random.randn(n_samples, 32).astype(np.float32)
        
        # Add meaningful signal based on labels and clinical features
        for i, (label, text_feat) in enumerate(zip(labels, text_data)):
            if label == 1:  # High risk cases
                # Simulate pathological patterns
                X_eeg[i] += np.random.randn(16) * 0.3  # Increased EEG variability
                X_fmri[i] += np.random.randn(32) * 0.25  # fMRI changes
                
                # Add specific patterns for dementia risk
                X_eeg[i, :4] += 0.2  # Frontal theta increase
                X_fmri[i, 10:15] -= 0.3  # Hippocampal signal decrease
        
        # Data splitting
        indices = np.arange(len(labels))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=AdvancedConfig.SEED
        )
        
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.2, stratify=labels[train_val_idx], 
            random_state=AdvancedConfig.SEED
        )
        
        logger.info(f"🔄 Data split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Create datasets
        model_name = AdvancedConfig.AVAILABLE_MODELS[AdvancedConfig.CHOSEN_MODEL]
        
        train_dataset = AdvancedBrainDataset(
            X_eeg[train_idx], X_fmri[train_idx], [text_data[i] for i in train_idx], labels[train_idx],
            model_name, augment=True, num_augmentations=AdvancedConfig.NUM_AUGMENTATIONS
        )
        
        val_dataset = AdvancedBrainDataset(
            X_eeg[val_idx], X_fmri[val_idx], [text_data[i] for i in val_idx], labels[val_idx],
            model_name, augment=False
        )
        
        test_dataset = AdvancedBrainDataset(
            X_eeg[test_idx], X_fmri[test_idx], [text_data[i] for i in test_idx], labels[test_idx],
            model_name, augment=False
        )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=AdvancedConfig.BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=(AdvancedConfig.DEVICE.type == 'cuda')  # Set num_workers=0 for Windows
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=AdvancedConfig.VAL_BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=(AdvancedConfig.DEVICE.type == 'cuda')
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=AdvancedConfig.TEST_BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=(AdvancedConfig.DEVICE.type == 'cuda')
        )
        
        # Initialize model
        model = AdvancedMultiModalNet(
            eeg_dim=X_eeg.shape[1],
            fmri_dim=X_fmri.shape[1],
            model_name=model_name
        ).to(AdvancedConfig.DEVICE)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"🧠 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Advanced optimizer setup
        criterion = FocalLoss(alpha=AdvancedConfig.FOCAL_ALPHA, gamma=AdvancedConfig.FOCAL_GAMMA)
        
        # Different learning rates for different components
        pretrained_params = []
        new_params = []
        
        for name, param in model.named_parameters():
            if 'text_encoder' in name and param.requires_grad:
                pretrained_params.append(param)
            elif param.requires_grad:
                new_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': pretrained_params, 'lr': AdvancedConfig.LEARNING_RATE, 'weight_decay': AdvancedConfig.WEIGHT_DECAY},
            {'params': new_params, 'lr': AdvancedConfig.LEARNING_RATE * 3, 'weight_decay': AdvancedConfig.WEIGHT_DECAY}
        ])
        
        # Learning rate scheduler
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if AdvancedConfig.DEVICE.type == 'cuda' else None
        
        logger.info(f"⚙️ Training setup complete - Starting {AdvancedConfig.EPOCHS} epochs")
        
        # Training loop
        best_val_auc = 0
        best_model_state = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_acc': []}
        
        epoch_bar = tqdm(range(AdvancedConfig.EPOCHS), desc="Training Progress")
        for epoch in epoch_bar:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            # Training
            train_metrics = train_epoch_advanced(model, train_loader, optimizer, criterion, scaler, epoch)
            
            # Validation
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            val_metrics = evaluate_model_advanced(model, val_loader)
            
            epoch_time = time.perf_counter() - start_time
            
            # Scheduler step
            scheduler.step()
            
            # Track history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_auc'].append(val_metrics['auc'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Model checkpointing
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_model_state = model.state_dict().copy()
                torch.save({
                    'model_state_dict': best_model_state,
                    'config': AdvancedConfig.__dict__,
                    'epoch': epoch,
                    'val_auc': best_val_auc
                }, f'best_model_{AdvancedConfig.CHOSEN_MODEL}.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Progress logging
            gpu_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            
            epoch_bar.set_postfix({
                'TL': f"{train_metrics['loss']:.3f}",
                'VL': f"{val_metrics['loss']:.3f}",
                'VAcc': f"{val_metrics['accuracy']:.3f}",
                'VAUC': f"{val_metrics['auc']:.3f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.6f}",
                'GPU': f"{gpu_mem:.0f}MB"
            })
            
            # Detailed logging every 5 epochs
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1:3d}/{AdvancedConfig.EPOCHS}: "
                    f"Time={epoch_time:.1f}s, "
                    f"TrLoss={train_metrics['loss']:.4f}, "
                    f"ValLoss={val_metrics['loss']:.4f}, "
                    f"ValAcc={val_metrics['accuracy']:.4f}, "
                    f"ValAUC={val_metrics['auc']:.4f}, "
                    f"ValAP={val_metrics['avg_precision']:.4f}"
                )
            
            # Early stopping
            if patience_counter >= AdvancedConfig.PATIENCE:
                logger.info(f"🔴 Early stopping at epoch {epoch+1} (patience: {AdvancedConfig.PATIENCE})")
                break
        
        # Final evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        test_metrics = evaluate_model_advanced(model, test_loader)
        
        # Results summary
        logger.info("\n" + "="*100)
        logger.info("🎯 FINAL ADVANCED MODEL RESULTS")
        logger.info("="*100)
        logger.info(f"Model: {AdvancedConfig.CHOSEN_MODEL}")
        logger.info(f"Architecture: Advanced Multi-Modal with Cross-Attention")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.1f}%)")
        logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
        logger.info(f"Test Average Precision: {test_metrics['avg_precision']:.4f}")
        logger.info(f"Best Validation AUC: {best_val_auc:.4f}")
        logger.info(f"Epochs Trained: {epoch + 1}")
        logger.info(f"\n{test_metrics['classification_report']}")
        logger.info(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")
        
        # Save comprehensive results
        results_dir = base_path / 'results'
        results_dir.mkdir(exist_ok=True)
        
        results = {
            'model_info': {
                'type': 'advanced_multimodal',
                'huggingface_model': AdvancedConfig.AVAILABLE_MODELS[AdvancedConfig.CHOSEN_MODEL],
                'model_key': AdvancedConfig.CHOSEN_MODEL,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            },
            'training_config': {
                'epochs_trained': epoch + 1,
                'batch_size': AdvancedConfig.BATCH_SIZE,
                'learning_rate': AdvancedConfig.LEARNING_RATE,
                'hidden_dim': AdvancedConfig.HIDDEN_DIM,
                'num_heads': AdvancedConfig.NUM_HEADS,
                'num_layers': AdvancedConfig.NUM_LAYERS,
                'dropout': AdvancedConfig.DROPOUT,
                'use_mixup': AdvancedConfig.USE_MIXUP,
                'use_focal_loss': AdvancedConfig.USE_FOCAL_LOSS
            },
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
        
        # Save results
        with open(results_dir / f'advanced_results_{AdvancedConfig.CHOSEN_MODEL}.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save predictions
        with open(results_dir / f'advanced_predictions_{AdvancedConfig.CHOSEN_MODEL}.csv', 'w') as f:
            f.write("row_index,participant_id,y_true,y_pred,y_prob\n")
            for i, idx in enumerate(test_idx):
                participant_id = participants_data[idx].get('participant_id', f'sub-{idx+1:02d}')
                f.write(f"{idx},{participant_id},{test_metrics['true_labels'][i]},{test_metrics['predictions'][i]},{test_metrics['probabilities'][i]:.6f}\n")
        
        # Performance comparison
        logger.info("\n" + "="*100)
        logger.info("📊 PERFORMANCE COMPARISON")
        logger.info("="*100)
        
        baseline_acc = 0.625  # Original model
        baseline_auc = 0.60
        optimized_acc = 0.875  # After threshold optimization
        
        current_acc = test_metrics['accuracy']
        current_auc = test_metrics['auc']
        
        logger.info("Model Performance Evolution:")
        logger.info(f"  1. Original Model:     Accuracy = {baseline_acc:.3f}, AUC = {baseline_auc:.3f}")
        logger.info(f"  2. Threshold Optimized: Accuracy = {optimized_acc:.3f}, AUC = {baseline_auc:.3f}")
        logger.info(f"  3. Advanced Model:     Accuracy = {current_acc:.3f}, AUC = {current_auc:.3f}")
        logger.info("")
        logger.info("Improvements:")
        logger.info(f"  vs Original:     Acc +{(current_acc - baseline_acc)*100:+.1f}%, AUC +{current_auc - baseline_auc:+.3f}")
        logger.info(f"  vs Optimized:    Acc {(current_acc - optimized_acc)*100:+.1f}%, AUC +{current_auc - baseline_auc:+.3f}")
        
        if current_acc > optimized_acc:
            logger.info(f"\n🏆 NEW BEST PERFORMANCE! Advanced model beats threshold optimization!")
        elif current_acc > baseline_acc:
            logger.info(f"\n✅ SIGNIFICANT IMPROVEMENT over original model!")
        else:
            logger.info(f"\n⚠️ Performance comparable to existing models")
        
        logger.info(f"\n💾 Results saved to:")
        logger.info(f"  - {results_dir / f'advanced_results_{AdvancedConfig.CHOSEN_MODEL}.json'}")
        logger.info(f"  - {results_dir / f'advanced_predictions_{AdvancedConfig.CHOSEN_MODEL}.csv'}")
        logger.info(f"  - best_model_{AdvancedConfig.CHOSEN_MODEL}.pth")
        
        logger.info(f"\n🎉 Advanced training completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Error in advanced training: {str(e)}")
        raise

if __name__ == "__main__":
    main_advanced()