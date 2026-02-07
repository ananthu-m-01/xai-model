"""
Enhanced Real Dataset Explainer with BiomedGPT Integration
=========================================================
Combines the existing model predictions with BiomedGPT-generated explanations.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# Import our existing explainer and the new BiomedGPT explainer
from real_dataset_explainer import RealDatasetExplainer
from biomedgpt_explainer import BiomedGPTExplainer

def enhanced_explanation_pipeline(
    participant_ids: Optional[list] = None,
    max_participants: int = 5,
    save_detailed_reports: bool = True
) -> Dict[str, Any]:
    """
    Complete pipeline that combines model predictions with BiomedGPT explanations.
    
    Args:
        participant_ids: Specific participant IDs to process (optional)
        max_participants: Maximum number of participants to process
        save_detailed_reports: Whether to save detailed individual reports
    
    Returns:
        Combined results with model predictions and BiomedGPT explanations
    """
    
    print("🧬 ENHANCED EXPLAINER PIPELINE")
    print("=" * 60)
    print("Combining BiomedBERT predictions with BiomedGPT explanations...")
    print()
    
    # Initialize explainers
    print("🔄 Initializing AI models...")
    model_explainer = RealDatasetExplainer()
    biomedgpt_explainer = BiomedGPTExplainer()
    
    # Load participant data
    participants_df = pd.read_csv("participants_with_labels.csv")
    print(f"✅ Loaded {len(participants_df)} participants from dataset")
    
    # Select participants to process
    if participant_ids:
        selected_participants = participants_df[
            participants_df['participant_id'].isin(participant_ids)
        ]
    else:
        # Select a balanced sample
        high_risk = participants_df[participants_df['class_label'] == 'High Risk'].head(max_participants//2)
        low_risk = participants_df[participants_df['class_label'] == 'Low Risk'].head(max_participants//2)
        selected_participants = pd.concat([high_risk, low_risk])
    
    print(f"📊 Processing {len(selected_participants)} participants...")
    print()
    
    enhanced_results = {
        'pipeline_info': {
            'model_explainer': 'BiomedBERT + Clinical Enhancement',
            'biomedgpt_explainer': biomedgpt_explainer.model_name,
            'biomedgpt_available': biomedgpt_explainer.model is not None,
            'total_processed': len(selected_participants),
            'timestamp': pd.Timestamp.now().isoformat()
        },
        'participant_explanations': []
    }
    
    # Process each participant
    for idx, (_, participant) in enumerate(selected_participants.iterrows(), 1):
        participant_id = participant['participant_id']
        actual_label = participant['class_label']
        
        print(f"🔍 Processing {idx}/{len(selected_participants)}: {participant_id}")
        print(f"   Actual Label: {actual_label}")
        
        try:
            # Step 1: Get model prediction and explanation
            print("   🤖 Generating BiomedBERT prediction...")
            model_explanation = model_explainer.explain_real_participant(participant_id)
            
            prediction = model_explanation['prediction']
            predicted_risk = prediction['risk_level']
            confidence = prediction['confidence']
            agreement = (predicted_risk == 'HIGH RISK' and actual_label == 'High Risk') or \
                       (predicted_risk == 'LOW RISK' and actual_label == 'Low Risk')
            
            print(f"   📊 Model Prediction: {predicted_risk} ({confidence:.1%} confidence)")
            print(f"   ✅ Agreement: {'✓' if agreement else '✗'}")
            
            # Step 2: Generate BiomedGPT explanations
            print("   🧠 Generating BiomedGPT explanations...")
            
            # Prepare data for BiomedGPT
            participant_data = participant.to_dict()
            model_output = {
                'predicted_risk': predicted_risk,
                'confidence': confidence,
                'actual_label': actual_label,
                'agreement': agreement
            }
            
            # Generate different types of explanations
            comprehensive_explanation = biomedgpt_explainer.generate_biomedgpt_explanation(
                patient_data=participant_data,
                model_output=model_output,
                explanation_type="comprehensive"
            )
            
            technical_explanation = biomedgpt_explainer.generate_biomedgpt_explanation(
                patient_data=participant_data,
                model_output=model_output,
                explanation_type="technical"
            )
            
            patient_friendly_explanation = biomedgpt_explainer.generate_biomedgpt_explanation(
                patient_data=participant_data,
                model_output=model_output,
                explanation_type="patient-friendly"
            )
            
            # Combine all explanations
            combined_explanation = {
                'participant_info': {
                    'participant_id': participant_id,
                    'age': participant.get('age'),
                    'sex': 'Female' if participant.get('sex') == '0' else 'Male' if participant.get('sex') == '1' else 'Unknown',
                    'education': participant.get('education'),
                    'actual_label': actual_label
                },
                'clinical_data': {
                    'cvlt_7': participant.get('CVLT_7'),
                    'rpm': participant.get('RPM'),
                    'bdi': participant.get('BDI'),
                    'dementia_history_parents': participant.get('dementia_history_parents'),
                    'apoe_haplotype': participant.get('APOE_haplotype'),
                    'total_cholesterol': participant.get('total_cholesterol'),
                    'cholesterol_HDL': participant.get('cholesterol_HDL')
                },
                'model_prediction': {
                    'risk_assessment': predicted_risk,
                    'confidence': confidence,
                    'agreement_with_label': agreement,
                    'biomedbert_analysis': model_explanation.get('for_doctor', {}).get('clinical_findings', [])
                },
                'biomedgpt_explanations': {
                    'comprehensive': comprehensive_explanation,
                    'technical': technical_explanation,
                    'patient_friendly': patient_friendly_explanation
                },
                'original_model_explanation': model_explanation
            }
            
            enhanced_results['participant_explanations'].append(combined_explanation)
            
            # Save individual detailed report if requested
            if save_detailed_reports:
                individual_file = f"results/detailed_explanation_{participant_id}.json"
                Path("results").mkdir(exist_ok=True)
                with open(individual_file, 'w') as f:
                    json.dump(combined_explanation, f, indent=2, default=str)
            
            print(f"   ✅ Complete explanation generated")
            print()
            
        except Exception as e:
            print(f"   ❌ Error processing {participant_id}: {e}")
            print()
            continue
    
    # Save combined results
    output_file = "results/enhanced_biomedgpt_explanations.json"
    Path("results").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(enhanced_results, f, indent=2, default=str)
    
    # Print summary
    print("📊 PIPELINE SUMMARY")
    print("=" * 40)
    print(f"Participants Processed: {len(enhanced_results['participant_explanations'])}")
    print(f"BiomedGPT Available: {'✅' if biomedgpt_explainer.model else '❌ (Using fallback)'}")
    print(f"Results Saved To: {output_file}")
    
    if save_detailed_reports:
        print(f"Individual Reports: results/detailed_explanation_[participant_id].json")
    
    # Calculate accuracy
    correct_predictions = sum(
        1 for exp in enhanced_results['participant_explanations']
        if exp['model_prediction']['agreement_with_label']
    )
    total_predictions = len(enhanced_results['participant_explanations'])
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    print(f"Model Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    print()
    
    return enhanced_results

def display_sample_explanation(results: Dict[str, Any], participant_index: int = 0):
    """Display a sample enhanced explanation in a readable format."""
    
    if not results['participant_explanations']:
        print("No explanations available to display")
        return
    
    explanation = results['participant_explanations'][participant_index]
    
    print("📋 SAMPLE ENHANCED EXPLANATION")
    print("=" * 50)
    
    # Basic info
    info = explanation['participant_info']
    print(f"Patient: {info['participant_id']}")
    print(f"Demographics: {info['age']}yo {info['sex']}, Education: {info['education']} years")
    print(f"Actual Risk Label: {info['actual_label']}")
    print()
    
    # Clinical data
    clinical = explanation['clinical_data']
    print("🔬 CLINICAL DATA:")
    print(f"  Memory (CVLT-7): {clinical['cvlt_7']}")
    print(f"  Fluid Intelligence (RPM): {clinical['rpm']}")
    print(f"  Depression (BDI): {clinical['bdi']}")
    print(f"  Family History: {clinical['dementia_history_parents']}")
    print(f"  APOE Genotype: {clinical['apoe_haplotype']}")
    print()
    
    # Model prediction
    prediction = explanation['model_prediction']
    print("🤖 BIOMEDBERT PREDICTION:")
    print(f"  Risk Assessment: {prediction['risk_assessment']}")
    print(f"  Confidence: {prediction['confidence']:.1%}")
    print(f"  Agrees with Label: {'✅' if prediction['agreement_with_label'] else '❌'}")
    print()
    
    # BiomedGPT explanations
    biomedgpt = explanation['biomedgpt_explanations']
    
    print("🧠 BIOMEDGPT PATIENT-FRIENDLY EXPLANATION:")
    print("-" * 45)
    if 'enhanced_clinical_explanation' in biomedgpt['patient_friendly']:
        print(biomedgpt['patient_friendly']['enhanced_clinical_explanation']['reasoning'])
    elif 'biomedgpt_explanation' in biomedgpt['patient_friendly']:
        print(biomedgpt['patient_friendly']['biomedgpt_explanation']['raw_response'][:400] + "...")
    print()
    
    print("🩺 BIOMEDGPT TECHNICAL ANALYSIS:")
    print("-" * 35)
    if 'enhanced_clinical_explanation' in biomedgpt['technical']:
        print(biomedgpt['technical']['enhanced_clinical_explanation']['reasoning'])
    elif 'biomedgpt_explanation' in biomedgpt['technical']:
        print(biomedgpt['technical']['biomedgpt_explanation']['raw_response'][:400] + "...")
    print()

if __name__ == "__main__":
    # Run the enhanced pipeline
    print("🚀 Starting Enhanced Explanation Pipeline...")
    print()
    
    # Process a small sample first
    results = enhanced_explanation_pipeline(
        max_participants=6,  # Process 6 participants (3 high risk, 3 low risk)
        save_detailed_reports=True
    )
    
    if results['participant_explanations']:
        print("\n" + "="*60)
        display_sample_explanation(results, 0)  # Show first participant
        
        print(f"\n🎯 Enhanced explanations generated for {len(results['participant_explanations'])} participants!")
        print("📁 Check the 'results' folder for detailed individual reports")
    else:
        print("❌ No explanations were generated successfully")