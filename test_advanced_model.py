"""
Test the trained advanced model with comprehensive evaluation
"""
import json
import torch
import numpy as np
from pathlib import Path
import csv

# Import your model classes
from model_huggingface_fixed import (
    AdvancedConfig, AdvancedBrainDataset, AdvancedMultiModalNet, evaluate_model_advanced,
    manual_train_test_split, manual_confusion_matrix, manual_classification_report, manual_roc_auc
)

def load_trained_model(model_path, device):
    """Load the trained model safely."""
    print(f"Loading model from: {model_path}")

    # Create model instance
    model = AdvancedMultiModalNet(
        eeg_dim=AdvancedConfig.EEG_DIM if hasattr(AdvancedConfig, 'EEG_DIM') else 16,
        fmri_dim=AdvancedConfig.FMRI_DIM if hasattr(AdvancedConfig, 'FMRI_DIM') else 32,
        model_name=AdvancedConfig.CHOSEN_MODEL
    )

    # Load state dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    model.to(device)
    model.eval()
    return model

def load_test_data():
    """Load the same test data used during training."""
    base_path = Path(__file__).parent

    # Load participants data manually (avoid pandas)
    participants_file = base_path / 'participants_with_labels.csv'
    participants_data = []
    
    with open(participants_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            participants_data.append(row)
    
    # Convert to list of dicts and create labels
    participants_df = participants_data
    labels = []
    for row in participants_df:
        # Create labels (same as training)
        dementia_history = float(row.get('dementia_history_parents', '0').strip() or '0')
        cvlt_7 = float(row.get('CVLT_7', '13.5').strip() or '13.5')
        risk_label = 1 if (dementia_history > 0 or cvlt_7 < 13.5) else 0
        labels.append(risk_label)
    
    y = np.array(labels)
    indices = np.arange(len(y))

    # Same train/val/test split as training
    train_val_idx, test_idx = manual_train_test_split(
        indices, y, test_size=0.2, random_state=AdvancedConfig.SEED
    )

    # Create synthetic features (replace with real features when available)
    X_eeg = np.random.randn(len(participants_df), AdvancedConfig.EEG_DIM).astype(np.float32)
    X_fmri = np.random.randn(len(participants_df), AdvancedConfig.FMRI_DIM).astype(np.float32)
    
    # Extract text features
    text_data = []
    for row in participants_df:
        text_row = {
            'dementia_history_parents': row.get('dementia_history_parents', ''),
            'learning_deficits': row.get('learning_deficits', ''),
            'other_diseases': row.get('other_diseases', ''),
            'drugs': row.get('drugs', ''),
            'allergies': row.get('allergies', '')
        }
        text_data.append(text_row)

    # Test data
    X_test_eeg = X_eeg[test_idx]
    X_test_fmri = X_fmri[test_idx]
    text_test = [text_data[i] for i in test_idx]
    y_test = y[test_idx]

    return X_test_eeg, X_test_fmri, text_test, y_test, [participants_df[i] for i in test_idx]

def create_test_dataloader(X_eeg, X_fmri, text_data, labels):
    """Create test data loader."""
    test_dataset = AdvancedBrainDataset(
        X_eeg, X_fmri, text_data, labels, 
        model_name=AdvancedConfig.CHOSEN_MODEL, augment=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=AdvancedConfig.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=(AdvancedConfig.DEVICE.type == 'cuda')
    )

    return test_loader

def comprehensive_evaluation(model, test_loader, y_true, save_path=None):
    """Comprehensive model evaluation."""
    print("\n🔍 Running comprehensive evaluation...")

    # Get predictions
    results = evaluate_model_advanced(model, test_loader)

    # Convert to numpy for sklearn metrics
    y_pred = results['predictions']
    y_prob = results['probabilities']

    # Calculate comprehensive metrics
    metrics = {
        'accuracy': float(np.mean(y_pred == y_true)),
        'auc': float(manual_roc_auc(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        'avg_precision': 0.0,  # Manual implementation not available, set to 0
        'loss': float(results['loss']),
        'classification_report': manual_classification_report(y_true, y_pred),
        'confusion_matrix': manual_confusion_matrix(y_true, y_pred).tolist(),
        'model_info': {
            'device': str(AdvancedConfig.DEVICE),
            'text_model': AdvancedConfig.TRANSFORMER_MODEL,
            'batch_size': AdvancedConfig.TEST_BATCH_SIZE
        }
    }

    # Print results
    print(f"\n📊 Test Results:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   AUC: {metrics['auc']:.4f}")
    print(f"   Average Precision: {metrics['avg_precision']:.4f}")
    print(f"   Loss: {metrics['loss']:.4f}")

    print(f"\n📋 Classification Report:")
    report = manual_classification_report(y_true, y_pred)
    for cls, scores in report.items():
        if cls == 'accuracy':
            print(f"   {cls}: {scores:.4f}")
        else:
            print(f"   {cls}: precision={scores['precision']:.4f}, recall={scores['recall']:.4f}, f1={scores['f1-score']:.4f}, support={scores['support']}")

    print(f"\n🔢 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"   Predicted:  0     1")
    print(f"   Actual: 0  [{cm[0][0]:2d}  {cm[0][1]:2d}]")
    print(f"           1  [{cm[1][0]:2d}  {cm[1][1]:2d}]")

    # Save results if path provided
    if save_path:
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_metrics = convert_numpy_types(metrics)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        print(f"\n💾 Results saved to: {save_path}")

    return metrics, results

def test_individual_samples(model, test_loader, participants_df, num_samples=5):
    """Test individual samples and show detailed predictions."""
    print(f"\n🧪 Testing {num_samples} individual samples...")

    model.eval()
    samples_tested = 0

    with torch.no_grad():
        for batch_idx, (eeg, fmri, text_inputs, labels) in enumerate(test_loader):
            if samples_tested >= num_samples:
                break

            eeg = eeg.to(AdvancedConfig.DEVICE)
            fmri = fmri.to(AdvancedConfig.DEVICE)
            labels = labels.to(AdvancedConfig.DEVICE)

            text_inputs = {
                'input_ids': text_inputs['input_ids'].to(AdvancedConfig.DEVICE),
                'attention_mask': text_inputs['attention_mask'].to(AdvancedConfig.DEVICE)
            }

            # Get predictions
            outputs = model(eeg, fmri, text_inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            # Process each sample in batch
            for i in range(len(labels)):
                if samples_tested >= num_samples:
                    break

                true_label = labels[i].item()
                pred_label = preds[i].item()
                prob_class1 = probs[i, 1].item()

                # Get participant info
                sample_idx = batch_idx * AdvancedConfig.TEST_BATCH_SIZE + i
                if sample_idx < len(participants_df):
                    participant_id = participants_df[sample_idx].get('participant_id', f'sample_{sample_idx}')
                    print(f"\nSample {samples_tested + 1}: Participant {participant_id}")
                else:
                    print(f"\nSample {samples_tested + 1}:")

                print(f"   True Label: {'High Risk' if true_label == 1 else 'Low Risk'}")
                print(f"   Predicted: {'High Risk' if pred_label == 1 else 'Low Risk'}")
                print(f"   Confidence: {prob_class1:.4f}")
                print(f"   Correct: {'✅' if pred_label == true_label else '❌'}")

                samples_tested += 1

def main():
    print("🧠 Advanced Multi-Modal Brain Analysis Model Tester")
    print("=" * 60)

    # Check device
    print(f"Using device: {AdvancedConfig.DEVICE}")

    # Load test data
    print("\n📂 Loading test data...")
    try:
        X_test_eeg, X_test_fmri, text_test, y_test, participants_df = load_test_data()
        print(f"✅ Loaded {len(y_test)} test samples")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return

    # Create test data loader
    test_loader = create_test_dataloader(X_test_eeg, X_test_fmri, text_test, y_test)

    # Load trained model - try different possible model files
    possible_models = [
        'best_model_biomedbert.pth',
        'best_model_advanced.pth', 
        'best_model.pth',
        'results/best_model.pth'
    ]
    
    model_path = None
    for path_str in possible_models:
        path = Path(path_str)
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        print("❌ No trained model found. Available model files:")
        for path_str in possible_models:
            path = Path(path_str)
            if path.exists():
                print(f"  ✅ {path_str}")
            else:
                print(f"  ❌ {path_str} (not found)")
        return

    model = load_trained_model(model_path, AdvancedConfig.DEVICE)
    if model is None:
        return

    # Comprehensive evaluation
    results_path = Path("results_advanced/test_results_comprehensive.json")
    metrics, detailed_results = comprehensive_evaluation(
        model, test_loader, y_test, save_path=results_path
    )

    # Test individual samples
    test_individual_samples(model, test_loader, participants_df)

    # Example: Generate GPT-style explanation for the first test sample
    print("\n🧠 GPT-style Explanation for First Test Sample:")
    from model_huggingface_fixed import explain_prediction_gpt_style
    # Preprocess text sample using the model's tokenizer (keep tensors)
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        # Fallback: use AdvancedBrainDataset tokenizer
        ds = AdvancedBrainDataset(X_test_eeg, X_test_fmri, text_test, y_test, model_name=AdvancedConfig.CHOSEN_MODEL)
        tokenizer = ds.tokenizer

    text_str = " ".join([str(v) for v in text_test[0].values()])
    text_inputs = tokenizer(text_str, return_tensors='pt', truncation=True, padding='max_length', max_length=AdvancedConfig.MAX_TEXT_LENGTH)
    # Move tensors to device inside explanation function, so keep as torch tensors here
    # Use deterministic clinician-focused explanation (more reliable than generator)
    try:
        explanation = explain_prediction_gpt_style(
            model,
            X_test_eeg[0],
            X_test_fmri[0],
            text_inputs,
            AdvancedConfig.DEVICE,
            patient_info=participants_df[0],
            use_generator=False  # Use deterministic clinician note
        )
        print(explanation)
    except Exception as e:
        print(f"Explanation failed: {e}")
        print("Basic prediction info:")
        print(f"Predicted: {'High Risk' if 1 else 'Low Risk'}")
        print(f"Confidence: Unknown")

    # Summary
    print(f"\n{'='*60}")
    print("🎯 TESTING COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {AdvancedConfig.TRANSFORMER_MODEL}")
    print(f"Test Accuracy: {metrics['accuracy']:.1%}")
    print(f"Test AUC: {metrics['auc']:.3f}")
    print(f"Results saved to: {results_path}")

    # Performance assessment
    accuracy = metrics['accuracy']
    auc = metrics['auc']

    if accuracy >= 0.90 and auc >= 0.95:
        print("🏆 EXCELLENT: Production-ready performance!")
    elif accuracy >= 0.80 and auc >= 0.85:
        print("✅ VERY GOOD: Strong performance for medical application!")
    elif accuracy >= 0.70 and auc >= 0.75:
        print("👍 GOOD: Acceptable performance, consider further improvements")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Consider model enhancements or more data")

if __name__ == "__main__":
    main()