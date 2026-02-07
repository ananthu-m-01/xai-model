from __future__ import annotations

import os
import logging
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
import mne
import nibabel as nib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    SEED = 42
    # Add regularization parameters
    DROPOUT = 0.3  # Reduced from 0.5
    WEIGHT_DECAY = 1e-2  # Reduced from 5e-2
    # Add early stopping parameters
    PATIENCE = 10
    # Adjust batch sizes for small dataset
    NUM_AUGMENTATIONS = 8  # Increased from 5
    BATCH_SIZE = 16  # Increased due to more data
    VAL_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    HIDDEN_DIM = 128
    NUM_HEADS = 4
    DEVICE = torch.device('cpu')  # Force CPU usage for compatibility
    
    # Transformer configurations
    TRANSFORMER_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    MAX_TEXT_LENGTH = 512
    TEXT_FEATURES = [
        'dementia_history_parents',
        'learning_deficits',
        'other_diseases',
        'drugs',
        'allergies'
    ]
    TEXT_HIDDEN_DIM = 768  # BERT hidden dimension
    FUSION_HIDDEN_DIM = 256  # Dimension for fused features

    # BERT configurations
    BERT_MODEL_NAME = "bert-base-uncased"
    MAX_TEXT_LENGTH = 128
    TEXT_FEATURES = [
        'dementia_history_parents',
        'learning_deficits',
        'other_diseases',
        'drugs',
        'allergies'
    ]
    
    # Add class weights for imbalanced data
    CLASS_WEIGHTS = [1.0, 2.2]  # Adjusted based on class distribution
    
    # Data configurations
    EEG_FILE_PATTERN = "sub-{}_task-rest_eeg.eeg"
    FMRI_DIR_AP_PATTERN = "sub-{}_task-rest_dir-AP_bold.nii.gz"
    FMRI_DIR_PA_PATTERN = "sub-{}_task-rest_dir-PA_bold.nii.gz"
    
    # Adjust hyperparameters for smaller dataset
    BATCH_SIZE = 8  # Reduced from 16
    VAL_BATCH_SIZE = 8
    TEST_BATCH_SIZE = 8
    NUM_AUGMENTATIONS = 12  # Increased for better balance
    EPOCHS = 100  # Increased for better convergence
    LEARNING_RATE = 5e-4  # Reduced for stability
    WEIGHT_DECAY = 5e-3  # Adjusted for regularization
    DROPOUT = 0.2  # Reduced for small dataset

def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocess_eeg(raw_eeg: np.ndarray, sfreq: int = 256) -> np.ndarray:
    """Preprocess EEG data with improved error handling."""
    try:
        # Apply bandpass filter for different frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30)
        }
        
        features = []
        for band_name, (low, high) in bands.items():
            filtered = mne.filter.filter_data(raw_eeg.astype(np.float64), 
                                           sfreq, low, high)
            psd, _ = mne.time_frequency.psd_array_welch(filtered, 
                                                       sfreq, 
                                                       fmin=low, 
                                                       fmax=high)
            features.append(np.mean(psd, axis=1))
        
        return np.concatenate(features)
    
    except Exception as e:
        logger.error(f"Error in EEG preprocessing: {str(e)}")
        raise

def preprocess_fmri(fmri_data: nib.Nifti1Image) -> np.ndarray:
    """Preprocess fMRI data with improved error handling."""
    try:
        # Apply spatial smoothing
        smoothed = nib.processing.smooth_image(fmri_data, fwhm=6)
        data = smoothed.get_fdata()
        
        # Calculate statistical features
        features = np.concatenate([
            data.mean(axis=(0, 1, 2)),
            data.std(axis=(0, 1, 2)),
            np.percentile(data, [25, 50, 75], axis=(0, 1, 2))
        ])
        
        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features.reshape(-1, 1)).ravel()
        
        return features
    
    except Exception as e:
        logger.error(f"Error in fMRI preprocessing: {str(e)}")
        raise

class BrainDataset(Dataset):
    """Enhanced Dataset class with text, EEG, and fMRI data handling."""
    def __init__(self, eeg_features: np.ndarray, fmri_features: np.ndarray, text_data: pd.DataFrame, labels: np.ndarray, augment: bool = False, num_augmentations: int = 5):
        assert len(eeg_features) == len(fmri_features) == len(labels) == len(text_data), "All inputs must have the same length"
        
        self.original_eeg = torch.FloatTensor(eeg_features)
        self.original_fmri = torch.FloatTensor(fmri_features)
        self.original_labels = torch.LongTensor(labels)
        self.text_data = text_data
        self.augment = augment
        self.num_augmentations = num_augmentations
        
        # Initialize tokenizer and process text
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TRANSFORMER_MODEL)
        self.processed_text = self._process_text()
        
        # Set up augmented data if needed
        if augment:
            self.eeg, self.fmri, self.labels = self._create_augmented_dataset()
        else:
            self.eeg = self.original_eeg
            self.fmri = self.original_fmri
            self.labels = self.original_labels
    
    def _process_text(self) -> list:
        processed = []
        for _, row in self.text_data.iterrows():
            text_parts = []
            for field in Config.TEXT_FEATURES:
                text_parts.append(f"{field}: {str(row.get(field, ''))}")
            text = " [SEP] ".join(text_parts)
            
            encoded = self.tokenizer(
                text,
                max_length=Config.MAX_TEXT_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            processed.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            })
        return processed
    
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
    def __init__(self, eeg_features: np.ndarray, fmri_features: np.ndarray, 
                 text_data: pd.DataFrame, labels: np.ndarray, 
                 augment: bool = False, num_augmentations: int = 5):
        """Initialize the dataset with EEG, fMRI, text, and label data."""
        assert len(eeg_features) == len(fmri_features) == len(labels) == len(text_data), \
            "All inputs must have the same length"
        
        self.original_eeg = torch.FloatTensor(eeg_features)
        self.original_fmri = torch.FloatTensor(fmri_features)
        self.original_labels = torch.LongTensor(labels)
        self.text_data = text_data
        self.augment = augment
        self.num_augmentations = num_augmentations
        
        # Initialize tokenizer and process text
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TRANSFORMER_MODEL)
        self.processed_text = self._process_text()
        
        # Set up augmented data if needed
        if augment:
            self.eeg, self.fmri, self.labels = self._create_augmented_dataset()
        else:
            self.eeg = self.original_eeg
            self.fmri = self.original_fmri
            self.labels = self.original_labels
            
        self.original_eeg = torch.FloatTensor(eeg_features)
        self.original_fmri = torch.FloatTensor(fmri_features)
        self.original_labels = torch.LongTensor(labels)
        self.text_data = text_data
        self.augment = augment
        self.num_augmentations = num_augmentations
        
        # Process text data
        self._initialize_text_processing()
        
        self.original_eeg = torch.FloatTensor(eeg_features)
        self.original_fmri = torch.FloatTensor(fmri_features)
        self.original_labels = torch.LongTensor(labels)
        self.text_data = text_data
        self.augment = augment
        self.num_augmentations = num_augmentations
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TRANSFORMER_MODEL)
        
        # Preprocess text data
        self.processed_text = self._preprocess_text()
        
        # Initialize tokenizer and process text
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TRANSFORMER_MODEL)
        self.text_data = text_data
        self.processed_text = self._preprocess_text()
        
        if augment:
            self.eeg, self.fmri, self.processed_text, self.labels = self._create_augmented_dataset()
        else:
            self.eeg = self.original_eeg
            self.fmri = self.original_fmri
            self.labels = self.original_labels
        
    def _create_augmented_dataset(self):
        """Create augmented samples using multiple techniques."""
        augmented_eeg = [self.original_eeg]
        augmented_fmri = [self.original_fmri]
        augmented_text = self.processed_text.copy()
        augmented_labels = [self.original_labels]
        
        for _ in range(self.num_augmentations):
            # Gaussian noise
            eeg_noise = self.original_eeg + torch.randn_like(self.original_eeg) * 0.1
            fmri_noise = self.original_fmri + torch.randn_like(self.original_fmri) * 0.1
            
            # Scaling
            eeg_scale = self.original_eeg * (1 + torch.randn_like(self.original_eeg) * 0.1)
            fmri_scale = self.original_fmri * (1 + torch.randn_like(self.original_fmri) * 0.1)
            
            # Time warping simulation
            eeg_warp = self._time_warp(self.original_eeg)
            fmri_warp = self._time_warp(self.original_fmri)
            
            # Feature masking
            eeg_mask = self._feature_mask(self.original_eeg)
            fmri_mask = self._feature_mask(self.original_fmri)
            
            # Add all augmentations
            augmented_eeg.extend([eeg_noise, eeg_scale, eeg_warp, eeg_mask])
            augmented_fmri.extend([fmri_noise, fmri_scale, fmri_warp, fmri_mask])
            augmented_text.extend(self.processed_text * 4)  # Repeat text for each augmentation
            augmented_labels.extend([self.original_labels] * 4)
        
        return (torch.cat(augmented_eeg, dim=0),
                torch.cat(augmented_fmri, dim=0),
                augmented_text,
                torch.cat(augmented_labels, dim=0))
    
    @staticmethod
    def _time_warp(data: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        """Simulate time warping by adding small shifts."""
        batch_size = data.shape[0]
        warped = data.clone()
        shifts = torch.randn(batch_size) * sigma
        for i in range(batch_size):
            if shifts[i] > 0:
                warped[i] = torch.roll(data[i], 1)
            else:
                warped[i] = torch.roll(data[i], -1)
        return warped
    
    @staticmethod
    def _feature_mask(data: torch.Tensor, mask_prob: float = 0.1) -> torch.Tensor:
        """Randomly mask features."""
        mask = torch.rand_like(data) > mask_prob
        return data * mask.float()
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def _preprocess_text(self) -> list:
        """Process text data and cache tokenized outputs."""
        processed = []
        for _, row in self.text_data.iterrows():
            # Combine all text fields with field names for context
            text_parts = []
            for field in Config.TEXT_FEATURES:
                text_parts.append(f"{field}: {str(row.get(field, ''))}")
            text = " [SEP] ".join(text_parts)
            
            # Tokenize text
            encoded = self.tokenizer(
                text,
                max_length=Config.MAX_TEXT_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Store as dictionary
            processed.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            })
            
        return processed

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict, torch.Tensor]:
        return (
            self.eeg[idx], 
            self.fmri[idx], 
            {
                'input_ids': self.processed_text[idx]['input_ids'].to(Config.DEVICE),
                'attention_mask': self.processed_text[idx]['attention_mask'].to(Config.DEVICE)
            },
            self.labels[idx]
        )

class DualStreamNetwork(nn.Module):
    """Enhanced multi-stream network with text, EEG, and fMRI processing."""
    def __init__(self, eeg_dim: int, fmri_dim: int, hidden_dim: int = Config.HIDDEN_DIM):
        super().__init__()
        
        # Initialize encoders for each modality
        self.eeg_encoder = self._create_encoder(eeg_dim, hidden_dim)
        self.fmri_encoder = self._create_encoder(fmri_dim, hidden_dim)
        
        # Initialize text encoder (BERT)
        self.text_encoder = AutoModel.from_pretrained(Config.TRANSFORMER_MODEL)
        
        # Text projection layer to match hidden dimension
        self.text_projector = nn.Linear(768, hidden_dim)  # BERT base has 768 dim
        
        # Freeze BERT parameters (optional, can be fine-tuned if needed)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # Text projection layer
        self.text_projector = nn.Linear(Config.TEXT_HIDDEN_DIM, hidden_dim)
        
        # Multi-head attention for modality fusion
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=Config.NUM_HEADS,
            dropout=Config.DROPOUT
        )
        
        # Enhanced classifier with modality fusion
        fusion_dim = hidden_dim * 3  # EEG + fMRI + Text
        self.fusion_layer = nn.Linear(fusion_dim, hidden_dim)
        self.classifier = self._create_classifier(hidden_dim)
        self._init_weights()
        
    @staticmethod
    def _create_encoder(input_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT)
        )
    
    @staticmethod
    def _create_classifier(hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(hidden_dim, 2)
        )
    
    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, eeg: torch.Tensor, fmri: torch.Tensor, text_inputs: dict) -> torch.Tensor:
        # Process EEG and fMRI
        eeg_features = self.eeg_encoder(eeg)
        fmri_features = self.fmri_encoder(fmri)
        
        # Process text with BERT and get pooled output
        text_outputs = self.text_encoder(**text_inputs)
        text_features = text_outputs.last_hidden_state.mean(dim=1)  # Mean pooling across sequence
        text_features = self.text_projector(text_features)
        
        # Reshape all features for attention
        eeg_features = eeg_features.unsqueeze(0)
        fmri_features = fmri_features.unsqueeze(0)
        text_features = text_features.unsqueeze(0)
        
        # Project features to same dimension
        text_features = self.text_projector(text_features)
        
        # Multi-modal attention
        attn_output_eeg, _ = self.attention(eeg_features, fmri_features, text_features)
        attn_output_fmri, _ = self.attention(fmri_features, eeg_features, text_features)
        attn_output_text, _ = self.attention(text_features, eeg_features, fmri_features)
        
        # Combine all modalities
        combined = torch.cat([
            attn_output_eeg.squeeze(0),
            attn_output_fmri.squeeze(0),
            attn_output_text.squeeze(0)
        ], dim=1)
        
        # Project to classifier input dimension
        combined = F.relu(self.fusion_layer(combined))
        
        # Final classification        return self.classifier(combined)

def train_epoch(model: nn.Module, data_loader: DataLoader, optimizer, criterion) -> dict:
    """Run one training epoch and return metrics."""
    model.train()
    running_loss = 0.0
    predictions = []
    true_labels = []

    for eeg, fmri, text_inputs, labels in data_loader:
        eeg = eeg.to(Config.DEVICE)
        fmri = fmri.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        # Move text inputs to device
        text_inputs = {
            'input_ids': text_inputs['input_ids'].to(Config.DEVICE),
            'attention_mask': text_inputs['attention_mask'].to(Config.DEVICE)
        }
        
        optimizer.zero_grad()
        outputs = model(eeg, fmri, text_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(data_loader.dataset)
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': true_labels
    }

def train_model(model: nn.Module, train_loader: DataLoader, 
                val_loader: DataLoader, epochs: int = Config.EPOCHS) -> dict:
    """Enhanced training function with early stopping and better regularization."""
    try:
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Add learning rate warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=Config.LEARNING_RATE,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
        
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(Config.CLASS_WEIGHTS).to(Config.DEVICE)
        )
        
        early_stopping = EarlyStopping()
        best_val_auc = 0
        best_model_state = None
        history = {
            'train_loss': [], 
            'train_accuracy': [], 
            'val_loss': [],
            'val_accuracy': [], 
            'val_auc': []
        }
        
        for epoch in range(epochs):
            # Training phase with gradient clipping
            model.train()
            train_metrics = train_epoch(model, train_loader, optimizer, criterion)
            
            # Validation phase
            val_metrics = evaluate_model(model, val_loader)
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_auc'].append(val_metrics['auc'])
            
            # Early stopping check
            early_stopping(val_metrics['loss'])
            if early_stopping.should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_model_state = model.state_dict()
                torch.save(best_model_state, 'best_model.pth')
            
            scheduler.step()
            
            logger.info(
                f'Epoch {epoch+1}/{epochs}: '
                f'Train Loss: {train_metrics["loss"]:.4f}, '
                f'Val Loss: {val_metrics["loss"]:.4f}, '
                f'Train Acc: {train_metrics["accuracy"]:.4f}, '
                f'Val Acc: {val_metrics["accuracy"]:.4f}, '
                f'Val AUC: {val_metrics["auc"]:.4f}'
            )
        
        return {'best_model_state': best_model_state, 'history': history}
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

class EarlyStopping:
    def __init__(self, patience=Config.PATIENCE, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def clean_text(text):
    """Clean and standardize text data."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = str(text).lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters but keep important medical punctuation
    text = text.replace('/', ' / ').replace('-', ' - ')
    return text

def format_text_features(df, text_features):
    """Format text features from DataFrame."""
    formatted_df = df.copy()
    
    for feature in text_features:
        if feature not in formatted_df.columns:
            logger.warning(f"Text feature {feature} not found in DataFrame. Adding empty column.")
            formatted_df[feature] = ""
        formatted_df[feature] = formatted_df[feature].apply(clean_text)
    
    # Combine text features into a single field
    formatted_df['combined_text'] = formatted_df[text_features].apply(
        lambda x: " [SEP] ".join([str(val) for val in x if val]), axis=1
    )
    
    return formatted_df

def load_dataset_info():
    """Load dataset information and create risk labels with text preprocessing"""
    base_path = Path(__file__).parent
    
    # Load participants JSON for metadata
    with open(base_path / 'participants.json', 'r') as f:
        participants_info = json.load(f)
    
    # Load participants CSV
    participants_df = pd.read_csv(base_path / 'participants_with_labels.csv')
    
    # Create risk labels based on clinical criteria
    participants_df['risk_label'] = ((participants_df['dementia_history_parents'].astype(float).fillna(0) > 0) | 
                                   (participants_df['CVLT_7'].astype(float).fillna(13.5) < 13.5)).astype(int)
    
    # Format text features
    participants_df = format_text_features(participants_df, Config.TEXT_FEATURES)
    
    return participants_info, participants_df

def load_brain_data(subject_id: str, data_type: str) -> np.ndarray:
    """Load brain imaging data for a subject"""
    base_path = Path(__file__).parent
    subject_path = base_path / f'sub-{subject_id}'
    
    if data_type == 'eeg':
        # Look for EEG data in subject's directory
        eeg_paths = list(subject_path.glob('**/eeg/*.set'))  # Adjust pattern based on actual files
        if eeg_paths:
            # Load EEG data using MNE
            raw = mne.io.read_raw_eeglab(eeg_paths[0], preload=True)
            data = raw.get_data()
            return preprocess_eeg(data)
    
    elif data_type == 'fmri':
        # Look for fMRI data in subject's directory
        fmri_paths = list(subject_path.glob('**/func/*bold.nii.gz'))  # Adjust pattern based on actual files
        if fmri_paths:
            # Load fMRI data using nibabel
            img = nib.load(fmri_paths[0])
            data = img.get_fdata()
            return preprocess_fmri(data)
    
    # Return empty array if data not found
    return np.array([])

def prepare_dataset(participants_df: pd.DataFrame) -> tuple:
    """Prepare dataset from available data including text features"""
    
    # Extract clinical features (excluding text features)
    clinical_features = [
        'age', 'CVLT_7', 'RPM', 'BDI', 
        'total_cholesterol', 'education', 'BMI'
    ]
    
    # Prepare clinical data
    X_clinical = participants_df[clinical_features].fillna(0).values
    scaler = StandardScaler()
    X_clinical_scaled = scaler.fit_transform(X_clinical)
    
    # Initialize tokenizer here for text processing
    tokenizer = AutoTokenizer.from_pretrained(Config.TRANSFORMER_MODEL)
    
    # Prepare text data
    text_df = participants_df[Config.TEXT_FEATURES]
    
    # Get labels
    y = participants_df['risk_label'].values
    
    # Create simple feature vectors for subjects without imaging data
    n_subjects = len(participants_df)
    X_eeg = np.zeros((n_subjects, 32))  # Assuming 32 EEG channels
    X_fmri = np.zeros((n_subjects, 64))  # Assuming 64 fMRI features
    
    # Try to load imaging data where available
    for idx, subject_id in enumerate(participants_df['participant_id']):
        eeg_data = load_brain_data(subject_id, 'eeg')
        fmri_data = load_brain_data(subject_id, 'fmri')
        
        if eeg_data.size > 0:
            X_eeg[idx] = eeg_data[:32]  # Take first 32 features
        if fmri_data.size > 0:
            X_fmri[idx] = fmri_data[:64]  # Take first 64 features
    
    logger.info(f"Dataset prepared with shapes:")
    logger.info(f"EEG: {X_eeg.shape}")
    logger.info(f"fMRI: {X_fmri.shape}")
    logger.info(f"Clinical: {X_clinical_scaled.shape}")
    logger.info(f"Text features: {len(Config.TEXT_FEATURES)} columns")
    logger.info(f"Labels: {y.shape}")
    
    return X_eeg, X_fmri, X_clinical_scaled, text_df, y
def evaluate_model(model: nn.Module, data_loader: DataLoader) -> dict:
    """Evaluate model performance"""
    model.eval()
    predictions = []
    true_labels = []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.2]).to(Config.DEVICE))
    
    with torch.no_grad():
        for eeg, fmri, text_inputs, labels in data_loader:
            try:
                eeg = eeg.to(Config.DEVICE)
                fmri = fmri.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                # Move text inputs to device
                text_inputs = {
                    'input_ids': text_inputs['input_ids'].to(Config.DEVICE),
                    'attention_mask': text_inputs['attention_mask'].to(Config.DEVICE)
                }
            
            outputs = model(eeg, fmri, text_inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    avg_loss = running_loss / len(data_loader.dataset)
    
    return {
        'loss': avg_loss,
        'accuracy': np.mean(predictions == true_labels),
        'auc': roc_auc_score(true_labels, predictions),
        'predictions': predictions,
        'true_labels': true_labels,
        'classification_report': classification_report(true_labels, predictions),
        'confusion_matrix': confusion_matrix(true_labels, predictions)
    }

def save_results(training_results: dict, test_metrics: dict):
    """Save training and test results"""
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Save training history
    torch.save(training_results['history'], 
              results_dir / 'training_history.pth')
    
    # Save test metrics
    metrics_to_save = {
        'accuracy': float(test_metrics['accuracy']),
        'auc': float(test_metrics['auc']),
        'classification_report': test_metrics['classification_report'],
        'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
        'predictions': test_metrics['predictions'].tolist(),
        'true_labels': test_metrics['true_labels'].tolist()
    }
    
    with open(results_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    
    # Save best model
    torch.save(training_results['best_model_state'], 
              results_dir / 'best_model.pth')

def main():
    """Main execution function with proper train/test separation."""
    try:
        set_seed(Config.SEED)
        
        # Load dataset information
        participants_info, participants_df = load_dataset_info()
        logger.info(f"Loaded data for {len(participants_df)} participants")
        
        # Prepare dataset
        X_eeg, X_fmri, X_clinical, text_df, y = prepare_dataset(participants_df)
        logger.info(f"Data shapes: EEG {X_eeg.shape}, fMRI {X_fmri.shape}, Clinical {X_clinical.shape}")
        
        # Add clinical features to EEG and fMRI data
        X_eeg = np.concatenate([X_eeg, X_clinical], axis=1)
        X_fmri = np.concatenate([X_fmri, X_clinical], axis=1)
        
        # First split: separate test set (20% of data)
        # This test set won't be touched until final evaluation
        indices = np.arange(len(y))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=y, random_state=Config.SEED
        )
        
        X_train_val_eeg, X_test_eeg = X_eeg[train_val_idx], X_eeg[test_idx]
        X_train_val_fmri, X_test_fmri = X_fmri[train_val_idx], X_fmri[test_idx]
        text_train_val, text_test = text_df.iloc[train_val_idx], text_df.iloc[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        
        # Second split: split remaining data into train and validation
        train_idx, val_idx = train_test_split(
            np.arange(len(y_train_val)), 
            test_size=0.2,
            stratify=y_train_val,
            random_state=Config.SEED
        )
        
        # Split all data types
        X_train_eeg, X_val_eeg = X_train_val_eeg[train_idx], X_train_val_eeg[val_idx]
        X_train_fmri, X_val_fmri = X_train_val_fmri[train_idx], X_train_val_fmri[val_idx]
        text_train = text_train_val.iloc[train_idx]
        text_val = text_train_val.iloc[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        # Create datasets with text data
        train_dataset = BrainDataset(X_train_eeg, X_train_fmri, text_train, y_train, 
                                   augment=True, 
                                   num_augmentations=Config.NUM_AUGMENTATIONS)
        
        val_dataset = BrainDataset(X_val_eeg, X_val_fmri, text_val, y_val, 
                                 augment=False)
        
        test_dataset = BrainDataset(X_test_eeg, X_test_fmri, text_test, y_test, 
                                  augment=False)
        
        # Log dataset sizes
        logger.info(f"Dataset splits:")
        logger.info(f"Training samples: {len(X_train_eeg)} (augmented to {len(train_dataset)})")
        logger.info(f"Validation samples: {len(X_val_eeg)}")
        logger.info(f"Test samples: {len(X_test_eeg)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, 
                                batch_size=Config.BATCH_SIZE,
                                shuffle=True)
        
        val_loader = DataLoader(val_dataset,
                              batch_size=Config.VAL_BATCH_SIZE,
                              shuffle=False)
        
        test_loader = DataLoader(test_dataset,
                               batch_size=Config.TEST_BATCH_SIZE,
                               shuffle=False)
        
        # Initialize model
        model = DualStreamNetwork(
            eeg_dim=X_train_eeg.shape[1],
            fmri_dim=X_train_fmri.shape[1]
        ).to(Config.DEVICE)
        
        # Train model using only train and validation sets
        results = train_model(model, train_loader, val_loader)
        
        # After training is complete, evaluate on the held-out test set
        model.load_state_dict(results['best_model_state'])
        test_metrics = evaluate_model(model, test_loader)
        logger.info(f"Final test metrics: {test_metrics}")
        
        # Save results
        save_results(results, test_metrics)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()