"""
Comprehensive test script demonstrating the integrated BiomedBERT explainability.
This script showcases the enhanced explainable AI capabilities with the biomedical model.
"""

import torch
import numpy as np
from pathlib import Path
import json

from biomedbert_explainer import BiomedBERTExplainer
from model_huggingface_fixed import AdvancedConfig

def test_single_patient_explanation():
    """Test explanation for a single patient with detailed BiomedBERT analysis."""
    print("\n🔬 Testing Single Patient BiomedBERT Explanation")
    print("=" * 60)
    
    # Initialize explainer with trained model
    model_path = 'best_model_biomedbert.pth'
    if not Path(model_path).exists():
        print(f"❌ Model file {model_path} not found")
        return
    
    explainer = BiomedBERTExplainer(model_path)
    
    if not explainer.model:
        print("❌ Failed to load model")
        return
    
    # Create synthetic test data representing a patient
    eeg_data = np.random.randn(AdvancedConfig.EEG_DIM).astype(np.float32)
    fmri_data = np.random.randn(AdvancedConfig.FMRI_DIM).astype(np.float32)
    
    # Create clinical text data with BiomedBERT tokenization
    clinical_text = (
        "PATIENT CLINICAL HISTORY: "
        "FAMILY HISTORY: Parental dementia history - Mother diagnosed with Alzheimer's disease at age 72 | "
        "COGNITIVE PROFILE: Learning difficulties documented - Reported difficulty with memory tasks | "
        "ASSESSMENT: Neuroimaging and electrophysiological evaluation for cognitive decline risk assessment. "
        "MODALITIES: Brain MRI functional connectivity analysis and EEG neural oscillation patterns."
    )
    
    # Tokenize using the model's tokenizer
    if hasattr(explainer.model, 'tokenizer'):
        tokenizer = explainer.model.tokenizer
        encoded = tokenizer(
            clinical_text,
            max_length=AdvancedConfig.MAX_TEXT_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        text_data = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
    else:
        print("❌ Tokenizer not available")
        return
    
    # Patient information
    participant_info = {
        'participant_id': 'test-001',
        'age': 68,
        'sex': 'F'
    }
    
    # Generate comprehensive explanation
    print("🧠 Generating comprehensive BiomedBERT explanation...")
    try:
        explanation = explainer.explain_prediction_enhanced(
            eeg_data=eeg_data,
            fmri_data=fmri_data,
            text_data=text_data,
            participant_info=participant_info,
            return_attention=True
        )
        
        # Display results
        print(f"\n📋 COMPREHENSIVE PATIENT EXPLANATION")
        print(f"Patient ID: {participant_info['participant_id']}")
        
        # Basic prediction
        print(f"\n1️⃣ Basic Clinical Explanation:")
        basic_pred = explanation.get('basic_prediction', 'Not available')
        print(f"{basic_pred}")
        
        # BiomedBERT analysis
        print(f"\n2️⃣ BiomedBERT Medical Analysis:")
        biomedbert_analysis = explanation.get('biomedbert_analysis', {})
        
        concepts = biomedbert_analysis.get('medical_concepts_detected', [])
        if concepts:
            print(f"   Medical Concepts Detected:")
            for concept in concepts[:5]:  # Show top 5
                print(f"     - {concept.get('concept', 'N/A')} ({concept.get('category', 'N/A')}) - {concept.get('relevance', 'N/A')} relevance")
        
        risk_indicators = biomedbert_analysis.get('risk_indicators', [])
        if risk_indicators:
            print(f"   Risk Indicators:")
            for risk in risk_indicators[:3]:  # Show top 3
                print(f"     - {risk.get('type', 'N/A')}: {risk.get('severity', 'N/A')} severity")
        
        relevance_score = biomedbert_analysis.get('biomedical_relevance_score', 0)
        print(f"   Biomedical Relevance Score: {relevance_score:.3f}")
        
        # Clinical risk factors
        print(f"\n3️⃣ Clinical Risk Factor Analysis:")
        risk_analysis = explanation.get('clinical_risk_factors', {})
        identified_factors = risk_analysis.get('identified_factors', [])
        if identified_factors:
            for factor in identified_factors:
                print(f"   - {factor.get('factor', 'N/A')} (Impact: {factor.get('impact', 'N/A')})")
                print(f"     {factor.get('description', 'N/A')}")
        
        risk_score = risk_analysis.get('risk_score', 0)
        print(f"   Overall Risk Score: {risk_score:.3f}")
        
        recommendations = risk_analysis.get('recommendations', [])
        if recommendations:
            print(f"   Clinical Recommendations:")
            for rec in recommendations:
                print(f"     • {rec}")
        
        # Feature importance
        print(f"\n4️⃣ Feature Importance Analysis:")
        importance = explanation.get('feature_importance', {})
        modality_contrib = importance.get('modality_contributions', {})
        if modality_contrib:
            print(f"   Modality Contributions:")
            for modality, contrib in modality_contrib.items():
                print(f"     - {modality.upper()}: {contrib:.1%}")
        
        # Confidence analysis
        print(f"\n5️⃣ Prediction Confidence Analysis:")
        confidence = explanation.get('confidence_analysis', {})
        pred_confidence = confidence.get('prediction_confidence', 0)
        reliability = confidence.get('reliability', 'unknown')
        print(f"   Prediction Confidence: {pred_confidence:.3f}")
        print(f"   Reliability Assessment: {reliability}")
        
        confidence_factors = confidence.get('confidence_factors', [])
        if confidence_factors:
            print(f"   Confidence Factors:")
            for factor in confidence_factors:
                print(f"     • {factor}")
        
        # Save detailed explanation
        output_file = Path('results/detailed_patient_explanation.json')
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(output_file, 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_explanation = _convert_tensors_to_lists(explanation)
            json.dump(json_explanation, f, indent=2)
        
        print(f"\n💾 Detailed explanation saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error generating explanation: {str(e)}")
        import traceback
        traceback.print_exc()

def _convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: _convert_tensors_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_tensors_to_lists(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def test_model_validation():
    """Test model validation and explainability integration."""
    print("\n🔍 Testing Model Validation & Explainability Integration")
    print("=" * 60)
    
    # Check if optimized predictions exist
    predictions_file = Path('results/test_predictions_optimized.csv')
    if not predictions_file.exists():
        print("❌ Optimized predictions not found. Running threshold optimization...")
        import subprocess
        try:
            result = subprocess.run(['python', 'optimize_threshold_fixed.py'], 
                                  capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                print("✅ Threshold optimization completed")
            else:
                print(f"❌ Threshold optimization failed: {result.stderr}")
                return
        except Exception as e:
            print(f"❌ Failed to run threshold optimization: {e}")
            return
    
    # Test BiomedBERT explainer cohort analysis
    explainer = BiomedBERTExplainer()
    
    print("🧠 Generating cohort-level BiomedBERT analysis...")
    try:
        report = explainer.generate_cohort_explanation_report(
            predictions_file,
            Path('results/biomedbert_cohort_report.json')
        )
        
        print(f"\n📊 COHORT ANALYSIS RESULTS")
        print(f"Cohort Size: {report.get('cohort_size', 'N/A')}")
        
        model_info = report.get('model_info', {})
        print(f"Model Architecture: {model_info.get('architecture', 'N/A')}")
        print(f"Text Encoder: {model_info.get('text_encoder', 'N/A')}")
        print(f"Modalities: {', '.join(model_info.get('modalities', []))}")
        
        performance = report.get('performance_summary', {})
        if performance:
            print(f"\nPerformance Summary:")
            print(f"  Accuracy: {performance.get('accuracy', 0):.3f}")
            print(f"  Total Predictions: {performance.get('total_predictions', 'N/A')}")
            print(f"  High Risk Predicted: {performance.get('high_risk_predicted', 'N/A')}")
        
        biomedbert_insights = report.get('biomedbert_insights', {})
        if biomedbert_insights:
            print(f"\nBiomedBERT Insights:")
            for key, value in biomedbert_insights.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        recommendations = report.get('clinical_recommendations', [])
        if recommendations:
            print(f"\nClinical Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        print(f"\n✅ Cohort analysis completed successfully")
        
    except Exception as e:
        print(f"❌ Error in cohort analysis: {str(e)}")

def display_model_summary():
    """Display summary of the integrated BiomedBERT model."""
    print("\n📋 BiomedBERT Model Integration Summary")
    print("=" * 60)
    
    print(f"🤖 Model Configuration:")
    print(f"   Chosen Model: {AdvancedConfig.CHOSEN_MODEL}")
    print(f"   Text Hidden Dim: {AdvancedConfig.TEXT_HIDDEN_DIM}")
    print(f"   Max Text Length: {AdvancedConfig.MAX_TEXT_LENGTH}")
    print(f"   EEG Dimensions: {AdvancedConfig.EEG_DIM}")
    print(f"   fMRI Dimensions: {AdvancedConfig.FMRI_DIM}")
    
    available_models = AdvancedConfig.AVAILABLE_MODELS
    print(f"\n🏥 Available Medical Models:")
    medical_models = ['biomedbert', 'clinicalbert', 'scibert', 'pubmedbert']
    for model_key in medical_models:
        if model_key in available_models:
            print(f"   ✅ {model_key}: {available_models[model_key]}")
    
    print(f"\n🧠 Explainability Features:")
    explainability_features = [
        "Gradient-based feature importance",
        "Cross-modal attention analysis", 
        "BiomedBERT concept extraction",
        "Clinical risk factor identification",
        "Prediction confidence assessment",
        "Medical domain knowledge integration"
    ]
    for feature in explainability_features:
        print(f"   ✅ {feature}")
    
    # Check for saved models
    print(f"\n💾 Available Trained Models:")
    model_files = [
        'best_model_biomedbert.pth',
        'best_model.pth', 
        'best_optimized_model.pth'
    ]
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"   ✅ {model_file}")
        else:
            print(f"   ❌ {model_file} (not found)")

def main():
    """Main test function."""
    print("🧬 BiomedBERT Explainability Integration Test")
    print("=" * 60)
    print("This script demonstrates the enhanced explainable AI capabilities")
    print("of the BiomedBERT-based multimodal brain imaging model.")
    
    # Display model summary
    display_model_summary()
    
    # Test model validation
    test_model_validation()
    
    # Test single patient explanation
    test_single_patient_explanation()
    
    print(f"\n🎉 BiomedBERT Explainability Integration Test Complete!")
    print(f"Check the 'results/' directory for generated explanation reports.")

if __name__ == "__main__":
    main()