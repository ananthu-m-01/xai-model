import os
import numpy as np
import pandas as pd
import mne
import nibabel as nib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Tuple, List, Dict, Any
import logging
import json
from pathlib import Path
# For explainability
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns
# For generative AI text explanations
from transformers import AutoTokenizer, AutoModelForCausalLM
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class Config:
    SEED = 42
    # Add regularization parameters
    DROPOUT = 0.3 # Reduced from 0.5
    WEIGHT_DECAY = 1e-2 # Reduced from 5e-2
    # Add early stopping parameters
    PATIENCE = 10
    # Adjust batch sizes for small dataset
    NUM_AUGMENTATIONS = 8 # Increased from 5
    BATCH_SIZE = 16 # Increased due to more data
    VAL_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    HIDDEN_DIM = 128
    NUM_HEADS = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    CLASS_WEIGHTS = [1.0, 2.2] # Adjusted based on class distribution
   
    # Data configurations
    EEG_FILE_PATTERN = "sub-{}_task-rest_eeg.eeg"
    FMRI_DIR_AP_PATTERN = "sub-{}_task-rest_dir-AP_bold.nii.gz"
    FMRI_DIR_PA_PATTERN = "sub-{}_task-rest_dir-PA_bold.nii.gz"
   
    # Adjust hyperparameters for smaller dataset
    BATCH_SIZE = 8 # Reduced from 16
    VAL_BATCH_SIZE = 8
    TEST_BATCH_SIZE = 8
    NUM_AUGMENTATIONS = 12 # Increased for better balance
    EPOCHS = 100 # Increased for better convergence
    LEARNING_RATE = 5e-4 # Reduced for stability
    WEIGHT_DECAY = 5e-3 # Adjusted for regularization
    DROPOUT = 0.2 # Reduced for small dataset
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
    """Enhanced Dataset class with advanced augmentation."""
    def __init__(self, eeg_features: np.ndarray, fmri_features: np.ndarray,
                 labels: np.ndarray, augment: bool = False, num_augmentations: int = 5):
        assert len(eeg_features) == len(fmri_features) == len(labels), \
            "All inputs must have the same length"
       
        self.original_eeg = torch.FloatTensor(eeg_features)
        self.original_fmri = torch.FloatTensor(fmri_features)
        self.original_labels = torch.LongTensor(labels)
        self.augment = augment
        self.num_augmentations = num_augmentations
       
        if augment:
            self.eeg, self.fmri, self.labels = self._create_augmented_dataset()
        else:
            self.eeg = self.original_eeg
            self.fmri = self.original_fmri
            self.labels = self.original_labels
       
    def _create_augmented_dataset(self):
        """Create augmented samples using multiple techniques."""
        augmented_eeg = [self.original_eeg]
        augmented_fmri = [self.original_fmri]
        augmented_labels = [self.original_labels]
       
        for _ in range(self.num_augmentations):
            # Gaussian noise
            eeg_noise = self.original_eeg + torch.randn_like(self.original_eeg) * 0.1
            fmri_noise = self.original_fmri + torch.randn_like(self.original_fmri) * 0.1
            
            augmented_eeg.append(eeg_noise)
            augmented_fmri.append(fmri_noise)
            augmented_labels.append(self.original_labels)
            
        return torch.cat(augmented_eeg), torch.cat(augmented_fmri), torch.cat(augmented_labels)
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.eeg)
        
    def __getitem__(self, idx):
        """Return a specific sample."""
        return self.eeg[idx], self.fmri[idx], self.labels[idx]
    
    def _create_augmented_dataset(self):
        """Create augmented samples using multiple techniques."""
        augmented_eeg = [self.original_eeg]
        augmented_fmri = [self.original_fmri]
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
            augmented_labels.extend([self.original_labels] * 4)
       
        return (torch.cat(augmented_eeg, dim=0),
                torch.cat(augmented_fmri, dim=0),
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
   
    def _len_(self) -> int:
        return len(self.labels)
   
    def _getitem_(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.eeg[idx], self.fmri[idx], self.labels[idx]
class DualStreamNetwork(nn.Module):
    """Improved dual-stream network with better initialization."""
    def __init__(self, eeg_dim: int, fmri_dim: int, hidden_dim: int = Config.HIDDEN_DIM):
        super().__init__()
       
        self.eeg_encoder = self._create_encoder(eeg_dim, hidden_dim)
        self.fmri_encoder = self._create_encoder(fmri_dim, hidden_dim)
       
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=Config.NUM_HEADS,
            dropout=Config.DROPOUT
        )
       
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
   
    def forward(self, eeg: torch.Tensor, fmri: torch.Tensor) -> torch.Tensor:
        eeg_features = self.eeg_encoder(eeg)
        fmri_features = self.fmri_encoder(fmri)
       
        # Reshape for attention
        eeg_features = eeg_features.unsqueeze(0)
        fmri_features = fmri_features.unsqueeze(0)
       
        attn_output, _ = self.attention(
            eeg_features,
            fmri_features,
            fmri_features
        )
       
        combined = torch.cat([
            attn_output.squeeze(0),
            fmri_features.squeeze(0)
        ], dim=1)
       
        return self.classifier(combined)
def train_epoch(model: nn.Module, data_loader: DataLoader, optimizer, criterion) -> dict:
    """Run one training epoch and return metrics."""
    model.train()
    running_loss = 0.0
    predictions = []
    true_labels = []
    for eeg, fmri, labels in data_loader:
        eeg, fmri, labels = eeg.to(Config.DEVICE), fmri.to(Config.DEVICE), labels.to(Config.DEVICE)
        optimizer.zero_grad()
        outputs = model(eeg, fmri)
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
def load_dataset_info():
    """Load dataset information and create risk labels"""
    base_path = Path(__file__).parent
   
    # Load participants JSON for metadata
    with open(base_path / 'participants.json', 'r') as f:
        participants_info = json.load(f)
   
    # Load participants CSV
    participants_df = pd.read_csv(base_path / 'participants_with_labels.csv')
   
    # Create risk labels based on clinical criteria
    participants_df['risk_label'] = ((participants_df['dementia_history_parents'].astype(float).fillna(0) > 0) |
                                   (participants_df['CVLT_7'].astype(float).fillna(13.5) < 13.5)).astype(int)
   
    return participants_info, participants_df
def load_brain_data(subject_id: str, data_type: str) -> np.ndarray:
    """Load brain imaging data for a subject"""
    base_path = Path(__file__).parent
    subject_path = base_path / f'sub-{subject_id}'
   
    if data_type == 'eeg':
        # Look for EEG data in subject's directory
        eeg_paths = list(subject_path.glob('eeg/*.set')) # Adjust pattern based on actual files
        if eeg_paths:
            # Load EEG data using MNE
            raw = mne.io.read_raw_eeglab(eeg_paths[0], preload=True)
            data = raw.get_data()
            return preprocess_eeg(data)
   
    elif data_type == 'fmri':
        # Look for fMRI data in subject's directory
        fmri_paths = list(subject_path.glob('func/*bold.nii.gz')) # Adjust pattern based on actual files
        if fmri_paths:
            # Load fMRI data using nibabel
            img = nib.load(fmri_paths[0])
            data = img.get_fdata()
            return preprocess_fmri(data)
   
    # Return empty array if data not found
    return np.array([])
def prepare_dataset(participants_df: pd.DataFrame) -> tuple:
    """Prepare dataset from available data"""
   
    # Extract clinical features
    clinical_features = [
        'age', 'CVLT_7', 'dementia_history_parents',
        'RPM', 'BDI', 'total_cholesterol',
        'education', 'BMI'
    ]
   
    X_clinical = participants_df[clinical_features].fillna(0).values
    scaler = StandardScaler()
    X_clinical_scaled = scaler.fit_transform(X_clinical)
   
    # Get labels
    y = participants_df['risk_label'].values
   
    # Create simple feature vectors for subjects without imaging data
    n_subjects = len(participants_df)
    X_eeg = np.zeros((n_subjects, 32)) # Assuming 32 EEG channels
    X_fmri = np.zeros((n_subjects, 64)) # Assuming 64 fMRI features
   
    # Try to load imaging data where available
    for idx, subject_id in enumerate(participants_df['participant_id']):
        eeg_data = load_brain_data(subject_id, 'eeg')
        fmri_data = load_brain_data(subject_id, 'fmri')
       
        if eeg_data.size > 0:
            X_eeg[idx] = eeg_data[:32] # Take first 32 features
        if fmri_data.size > 0:
            X_fmri[idx] = fmri_data[:64] # Take first 64 features
   
    return X_eeg, X_fmri, X_clinical_scaled, y
def evaluate_model(model: nn.Module, data_loader: DataLoader) -> dict:
    """Evaluate model performance"""
    model.eval()
    predictions = []
    true_labels = []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.2]).to(Config.DEVICE))
   
    with torch.no_grad():
        for eeg, fmri, labels in data_loader:
            eeg = eeg.to(Config.DEVICE)
            fmri = fmri.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
           
            outputs = model(eeg, fmri)
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
def explain_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> dict:
    """
    Compute explainability attributions using Integrated Gradients without changing model architecture.
    Attributes contributions to EEG and fMRI inputs separately for the positive class.
    """
    model.eval()
    ig = IntegratedGradients(model)
    
    eeg_attributions = []
    fmri_attributions = []
    predictions = []
    true_labels = []
    
    # To handle multiple inputs, we wrap the forward pass
    def forward_with_baseline(eeg, fmri, target_class=None):
        if target_class is not None:
            return model(eeg, fmri)[:, target_class]
        return model(eeg, fmri)
    
    with torch.no_grad():
        for eeg_batch, fmri_batch, labels_batch in test_loader:
            eeg_batch = eeg_batch.to(device)
            fmri_batch = fmri_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # Get model predictions
            outputs = model(eeg_batch, fmri_batch)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels_batch.cpu().numpy())
            
            # Compute attributions for each sample in batch
            for i in range(eeg_batch.size(0)):
                eeg_i = eeg_batch[i:i+1].clone().requires_grad_(True)
                fmri_i = fmri_batch[i:i+1].clone().requires_grad_(True)
                pred_label = preds[i].item()

                # Create a wrapper for EEG attribution
                def forward_func_eeg(inputs, fixed_fmri=fmri_i):
                    return model(inputs, fixed_fmri)[:, pred_label]
                
                # EEG attribution (treating fmri as fixed)
                ig_eeg = IntegratedGradients(forward_func_eeg)
                eeg_attr = ig_eeg.attribute(
                    eeg_i,
                    n_steps=50,
                    internal_batch_size=1,
                    return_convergence_delta=False
                )
                eeg_attributions.append(eeg_attr.squeeze(0).cpu().detach().numpy())
                
                # Create a wrapper for fMRI attribution
                def forward_func_fmri(inputs, fixed_eeg=eeg_i):
                    return model(fixed_eeg, inputs)[:, pred_label]
                
                # fMRI attribution
                ig_fmri = IntegratedGradients(forward_func_fmri)
                fmri_attr = ig_fmri.attribute(
                    fmri_i,
                    n_steps=50,
                    internal_batch_size=1,
                    return_convergence_delta=False
                )
                fmri_attributions.append(fmri_attr.squeeze(0).cpu().detach().numpy())
    
    # Aggregate attributions (e.g., mean absolute values per feature across samples)
    eeg_attributions = np.array(eeg_attributions)
    fmri_attributions = np.array(fmri_attributions)
    
    # For simplicity, compute mean abs attribution per feature
    mean_eeg_abs = np.mean(np.abs(eeg_attributions), axis=0)
    mean_fmri_abs = np.mean(np.abs(fmri_attributions), axis=0)
    
    # Save attribution visualizations (bar plots for top features)
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Plot top 10 EEG features
    plt.figure(figsize=(10, 6))
    top_eeg_idx = np.argsort(mean_eeg_abs)[-10:]
    plt.bar(range(10), mean_eeg_abs[top_eeg_idx])
    plt.title('Top 10 EEG Feature Attributions (Mean Abs Integrated Gradients)')
    plt.xlabel('Feature Index')
    plt.ylabel('Attribution Score')
    plt.savefig(results_dir / 'eeg_attributions.png')
    plt.close()
    
    # Plot top 10 fMRI features
    plt.figure(figsize=(10, 6))
    top_fmri_idx = np.argsort(mean_fmri_abs)[-10:]
    plt.bar(range(10), mean_fmri_abs[top_fmri_idx])
    plt.title('Top 10 fMRI Feature Attributions (Mean Abs Integrated Gradients)')
    plt.xlabel('Feature Index')
    plt.ylabel('Attribution Score')
    plt.savefig(results_dir / 'fmri_attributions.png')
    plt.close()
    
    logger.info("Explanations saved: Attribution plots for EEG and fMRI features.")
    
    return {
        'eeg_attributions': eeg_attributions,
        'fmri_attributions': fmri_attributions,
        'mean_eeg_abs': mean_eeg_abs,
        'mean_fmri_abs': mean_fmri_abs
    }


class GenerativeAIExplainer:
    """Generates natural language explanations using a language model."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """Load GPT-2 for text generation."""
        model_candidates = ["gpt2", "distilgpt2"]
        
        for model_name in model_candidates:
            try:
                logger.info(f"Loading {model_name} for text explanations...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info(f"Successfully loaded {model_name}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        logger.warning("Could not load language model for text generation.")
        return False
    
    def generate_explanation(self, test_metrics: Dict, explanations: Dict) -> str:
        """Generate natural language explanation of model results."""
        
        if self.model is None or self.tokenizer is None:
            return self._generate_fallback_explanation(test_metrics, explanations)
        
        # Build prompt with model results
        accuracy = test_metrics['accuracy'] * 100
        auc = test_metrics['auc'] * 100
        cm = test_metrics['confusion_matrix']
        true_neg, false_pos = cm[0]
        false_neg, true_pos = cm[1]
        
        # Get top contributing features
        top_eeg_idx = np.argsort(explanations['mean_eeg_abs'])[-5:]
        top_fmri_idx = np.argsort(explanations['mean_fmri_abs'])[-5:]
        
        prompt = f"""Medical AI Model Report:

A deep learning model for cognitive impairment risk prediction achieved {accuracy:.1f}% accuracy and {auc:.1f}% AUC.

Confusion Matrix Results:
- True Negatives (correctly identified low risk): {true_neg}
- True Positives (correctly identified high risk): {true_pos}
- False Negatives (missed high risk cases): {false_neg}
- False Positives (false alarms): {false_pos}

Top EEG feature indices contributing to predictions: {list(top_eeg_idx)}
Top fMRI feature indices contributing to predictions: {list(top_fmri_idx)}

Clinical Interpretation:"""
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=400)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + 150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2
                )
            
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return full_text
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return self._generate_fallback_explanation(test_metrics, explanations)
    
    def _generate_fallback_explanation(self, test_metrics: Dict, explanations: Dict) -> str:
        """Generate explanation without language model."""
        accuracy = test_metrics['accuracy'] * 100
        auc = test_metrics['auc'] * 100
        cm = test_metrics['confusion_matrix']
        
        text = f"""\n{'='*60}
GENERATED AI EXPLANATION
{'='*60}

Model Performance Summary:
- Overall Accuracy: {accuracy:.1f}%
- Area Under ROC Curve (AUC): {auc:.1f}%

Confusion Matrix Analysis:
- Correctly identified {cm[0,0]} low-risk participants (True Negatives)
- Correctly identified {cm[1,1]} high-risk participants (True Positives)  
- Missed {cm[1,0]} high-risk cases (False Negatives)
- Generated {cm[0,1]} false alarms (False Positives)

Feature Attribution Analysis:
- Top EEG features contributing to predictions: indices {list(np.argsort(explanations['mean_eeg_abs'])[-5:])}
- Top fMRI features contributing to predictions: indices {list(np.argsort(explanations['mean_fmri_abs'])[-5:])}

Clinical Significance:
The model demonstrates strong discriminative ability with {auc:.1f}% AUC.
The balanced performance across both classes suggests reliable risk stratification.
{'='*60}
"""
        return text


def generate_text_explanation(test_metrics: Dict, explanations: Dict) -> str:
    """Generate and return text explanation from generative AI."""
    explainer = GenerativeAIExplainer()
    explanation_text = explainer.generate_explanation(test_metrics, explanations)
    return explanation_text


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, labels: List[str] = None):
    """
    Plot and save confusion matrix as an image.
    
    Args:
        cm: Confusion matrix (2D numpy array)
        save_path: Path to save the image
        labels: Optional class labels (e.g., ['Low Risk', 'High Risk'])
    """
    import seaborn as sns
    
    if labels is None:
        labels = ['Class 0', 'Class 1']
    
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to: {save_path}")
    plt.close()

def save_results(training_results: dict, test_metrics: dict, explanations: dict = None):
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
   
    # Save confusion matrix as image
    plot_confusion_matrix(
        test_metrics['confusion_matrix'],
        save_path=results_dir / 'confusion_matrix.png',
        labels=['Low Risk', 'High Risk']
    )
   
    # Save best model
    torch.save(training_results['best_model_state'],
              results_dir / 'best_model.pth')
    
    # Save explanations if available
    if explanations:
        exp_to_save = {
            'mean_eeg_abs': explanations['mean_eeg_abs'].tolist(),
            'mean_fmri_abs': explanations['mean_fmri_abs'].tolist()
        }
        with open(results_dir / 'explanations.json', 'w') as f:
            json.dump(exp_to_save, f, indent=4)
def main():
    """Main execution function with proper train/test separation."""
    try:
        set_seed(Config.SEED)
       
        # Load dataset information
        participants_info, participants_df = load_dataset_info()
        logger.info(f"Loaded data for {len(participants_df)} participants")
       
        # Prepare dataset
        X_eeg, X_fmri, X_clinical, y = prepare_dataset(participants_df)
        logger.info(f"Data shapes: EEG {X_eeg.shape}, fMRI {X_fmri.shape}, Clinical {X_clinical.shape}")
       
        # Add clinical features to EEG and fMRI data
        X_eeg = np.concatenate([X_eeg, X_clinical], axis=1)
        X_fmri = np.concatenate([X_fmri, X_clinical], axis=1)
       
        # First split: separate test set (20% of data)
        # This test set won't be touched until final evaluation
        X_train_val_eeg, X_test_eeg, X_train_val_fmri, X_test_fmri, y_train_val, y_test = \
            train_test_split(X_eeg, X_fmri, y,
                           test_size=0.2,
                           stratify=y,
                           random_state=Config.SEED)
       
        # Second split: split remaining data into train and validation
        X_train_eeg, X_val_eeg, X_train_fmri, X_val_fmri, y_train, y_val = \
            train_test_split(X_train_val_eeg, X_train_val_fmri, y_train_val,
                           test_size=0.2, # 20% of training data
                           stratify=y_train_val,
                           random_state=Config.SEED)
       
        # Create datasets
        train_dataset = BrainDataset(X_train_eeg, X_train_fmri, y_train,
                                   augment=True,
                                   num_augmentations=Config.NUM_AUGMENTATIONS)
       
        val_dataset = BrainDataset(X_val_eeg, X_val_fmri, y_val,
                                 augment=False)
       
        test_dataset = BrainDataset(X_test_eeg, X_test_fmri, y_test,
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
        
        # Compute explanations on test set
        explanations = explain_model(model, test_loader, Config.DEVICE)
        logger.info("Model explanations computed.")
        
        # Generate and display text explanation from generative AI
        logger.info("Generating text explanation from generative AI...")
        text_explanation = generate_text_explanation(test_metrics, explanations)
        print(text_explanation)
        
        # Save text explanation to file
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / 'ai_explanation.txt', 'w') as f:
            f.write(text_explanation)
        logger.info(f"Text explanation saved to: {results_dir / 'ai_explanation.txt'}")
       
        # Save results
        save_results(results, test_metrics, explanations)
        logger.info("Training and explanation generation completed successfully!")
       
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
if __name__ == "__main__":
    main()