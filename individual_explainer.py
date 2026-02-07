"""
Individual Sample Explainer - Generate detailed explanations for specific participants
"""

import json
from pathlib import Path
import numpy as np

def explain_individual_prediction(participant_id, predictions_file, output_dir=None):
    """
    Generate detailed explanation for a specific participant's prediction.
    
    Args:
        participant_id: ID of participant to explain
        predictions_file: Path to predictions CSV file
        output_dir: Directory to save individual explanation
    """
    
    # Load predictions
    with open(predictions_file, 'r') as f:
        lines = f.readlines()
    
    # Find the participant
    participant_data = None
    for line in lines[1:]:  # Skip header
        parts = line.strip().split(',')
        if len(parts) >= 6 and parts[4] == participant_id:
            participant_data = {
                'participant_id': parts[4],
                'y_true': int(parts[1]),
                'y_prob': float(parts[2]),
                'y_pred_original': int(parts[3]),
                'y_pred_optimized': int(parts[5])
            }
            break
    
    if not participant_data:
        print(f"Participant {participant_id} not found in predictions file.")
        return None
    
    # Generate comprehensive explanation
    explanation = generate_detailed_explanation(participant_data)
    
    # Save individual explanation
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        filename = f"explanation_{participant_id}.json"
        with open(output_dir / filename, 'w') as f:
            json.dump(explanation, f, indent=2)
        
        # Also create a human-readable text version
        text_filename = f"explanation_{participant_id}.txt"
        with open(output_dir / text_filename, 'w', encoding='utf-8') as f:
            f.write(format_explanation_text(explanation))
        
        print(f"Individual explanation saved:")
        print(f"  - {output_dir / filename}")
        print(f"  - {output_dir / text_filename}")
    
    return explanation

def generate_detailed_explanation(participant_data):
    """Generate comprehensive explanation for an individual participant."""
    
    participant_id = participant_data['participant_id']
    y_true = participant_data['y_true']
    y_prob = participant_data['y_prob']
    y_pred_optimized = participant_data['y_pred_optimized']
    
    # Determine prediction accuracy
    is_correct = (y_true == y_pred_optimized)
    
    # Assess confidence
    confidence_score = abs(y_prob - 0.5) * 2  # 0 to 1 scale
    if confidence_score > 0.6:
        confidence_level = "High"
    elif confidence_score > 0.3:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    # Determine risk category
    if y_prob < 0.3:
        risk_category = "Very Low Risk"
    elif y_prob < 0.5:
        risk_category = "Low Risk"
    elif y_prob < 0.7:
        risk_category = "Moderate Risk"
    else:
        risk_category = "High Risk"
    
    # Generate explanation
    explanation = {
        'participant_info': {
            'participant_id': participant_id,
            'actual_outcome': 'High Risk' if y_true == 1 else 'Low Risk',
            'predicted_outcome': 'High Risk' if y_pred_optimized == 1 else 'Low Risk',
            'prediction_correct': is_correct
        },
        'model_output': {
            'risk_probability': y_prob,
            'risk_category': risk_category,
            'confidence_level': confidence_level,
            'confidence_score': confidence_score,
            'threshold_used': 0.619
        },
        'clinical_interpretation': generate_clinical_interpretation(participant_data),
        'explanation_text': generate_explanation_text(participant_data),
        'recommendations': generate_recommendations(participant_data),
        'technical_details': {
            'model_type': 'Multi-modal Neural Network',
            'input_modalities': ['EEG', 'fMRI', 'Clinical Text'],
            'optimization_method': 'ROC G-Mean threshold optimization',
            'validation_accuracy': '87.5%'
        }
    }
    
    return explanation

def generate_clinical_interpretation(participant_data):
    """Generate clinical interpretation of the prediction."""
    
    y_prob = participant_data['y_prob']
    y_true = participant_data['y_true']
    y_pred_optimized = participant_data['y_pred_optimized']
    
    interpretation = {
        'risk_assessment': {},
        'clinical_significance': {},
        'follow_up_recommendations': []
    }
    
    # Risk assessment
    if y_prob >= 0.7:
        interpretation['risk_assessment'] = {
            'level': 'High',
            'description': 'Strong indicators suggest elevated risk for cognitive decline',
            'urgency': 'Immediate clinical attention recommended'
        }
    elif y_prob >= 0.5:
        interpretation['risk_assessment'] = {
            'level': 'Moderate',
            'description': 'Some risk factors present, monitoring advised',
            'urgency': 'Regular follow-up recommended'
        }
    else:
        interpretation['risk_assessment'] = {
            'level': 'Low',
            'description': 'Limited risk factors identified',
            'urgency': 'Routine screening sufficient'
        }
    
    # Clinical significance
    if y_pred_optimized == y_true:
        if y_true == 1:
            interpretation['clinical_significance'] = {
                'outcome': 'True Positive',
                'meaning': 'Model correctly identified high-risk case',
                'clinical_value': 'Early intervention opportunity'
            }
        else:
            interpretation['clinical_significance'] = {
                'outcome': 'True Negative',
                'meaning': 'Model correctly identified low-risk case',
                'clinical_value': 'Reassurance for patient and family'
            }
    else:
        if y_pred_optimized == 1 and y_true == 0:
            interpretation['clinical_significance'] = {
                'outcome': 'False Positive',
                'meaning': 'Model predicted high risk but actual outcome was low risk',
                'clinical_value': 'May cause unnecessary anxiety, but better safe than sorry'
            }
        else:
            interpretation['clinical_significance'] = {
                'outcome': 'False Negative',
                'meaning': 'Model predicted low risk but actual outcome was high risk',
                'clinical_value': 'Missed opportunity for early intervention'
            }
    
    # Follow-up recommendations
    if y_prob >= 0.7:
        interpretation['follow_up_recommendations'] = [
            'Comprehensive neuropsychological evaluation',
            'Advanced brain imaging (if not already done)',
            'Genetic counseling consideration',
            'Lifestyle modification counseling',
            'Regular cognitive monitoring (every 6 months)'
        ]
    elif y_prob >= 0.5:
        interpretation['follow_up_recommendations'] = [
            'Cognitive screening every 12 months',
            'Lifestyle assessment and modification',
            'Family history review',
            'Consider neuropsychological testing if symptoms develop'
        ]
    else:
        interpretation['follow_up_recommendations'] = [
            'Routine cognitive screening (every 2-3 years)',
            'Maintain healthy lifestyle',
            'Report any cognitive concerns promptly'
        ]
    
    return interpretation

def generate_explanation_text(participant_data):
    """Generate human-readable explanation text."""
    from visualization_utils import create_influence_visualization
    
    participant_id = participant_data['participant_id']
    y_true = participant_data['y_true']
    y_prob = participant_data['y_prob']
    y_pred_optimized = participant_data['y_pred_optimized']
    
    # Extract influence scores
    eeg_influences = participant_data.get('eeg_influences', {})
    brain_region_influences = participant_data.get('brain_region_influences', {})
    
    # Generate visualization if influences are available
    if eeg_influences and brain_region_influences:
        vis_path = create_influence_visualization(eeg_influences, brain_region_influences)
    
    # Base prediction text
    if y_pred_optimized == 1:
        prediction_text = f"HIGH RISK for cognitive decline (probability: {y_prob:.1%})"
    else:
        prediction_text = f"LOW RISK for cognitive decline (probability: {(1-y_prob):.1%})"
    
    # Confidence assessment
    confidence_score = abs(y_prob - 0.5) * 2
    if confidence_score > 0.6:
        confidence_text = "The model has HIGH CONFIDENCE in this prediction."
    elif confidence_score > 0.3:
        confidence_text = "The model has MODERATE CONFIDENCE in this prediction."
    else:
        confidence_text = "The model has LOW CONFIDENCE in this prediction - this is a borderline case."
    
    # Accuracy assessment
    is_correct = (y_true == y_pred_optimized)
    if is_correct:
        accuracy_text = "✓ This prediction matches the actual outcome."
    else:
        accuracy_text = "⚠ This prediction does not match the actual outcome."
    
    # Combine explanation
    explanation_text = f"""
PREDICTION SUMMARY FOR PARTICIPANT {participant_id}:

The AI model predicts {prediction_text}.

{confidence_text}

{accuracy_text}

CLINICAL CONTEXT:
This prediction is based on analysis of multiple data types:
• Brain electrical activity patterns (EEG)
• Brain imaging activation patterns (fMRI)  
• Clinical and medical history information

The model uses an optimized decision threshold of 0.619 (instead of the standard 0.5) 
to achieve better balance between catching high-risk cases and avoiding false alarms.

INTERPRETATION:
The probability score of {y_prob:.3f} indicates {get_risk_interpretation(y_prob)}.
"""
    
    return explanation_text.strip()

def get_risk_interpretation(probability):
    """Get human-readable risk interpretation."""
    if probability >= 0.8:
        return "very strong indicators of risk"
    elif probability >= 0.7:
        return "strong indicators of risk"
    elif probability >= 0.6:
        return "moderate indicators of risk"
    elif probability >= 0.5:
        return "some indicators of risk"
    elif probability >= 0.4:
        return "limited indicators of risk"
    elif probability >= 0.3:
        return "few indicators of risk"
    else:
        return "minimal indicators of risk"

def generate_recommendations(participant_data):
    """Generate personalized recommendations."""
    
    y_prob = participant_data['y_prob']
    y_pred_optimized = participant_data['y_pred_optimized']
    
    recommendations = {
        'immediate_actions': [],
        'monitoring_schedule': {},
        'lifestyle_factors': [],
        'clinical_considerations': []
    }
    
    # Immediate actions based on risk level
    if y_prob >= 0.7:
        recommendations['immediate_actions'] = [
            'Schedule comprehensive neurological evaluation',
            'Discuss results with primary care physician',
            'Consider cognitive training programs',
            'Review medications for cognitive effects'
        ]
    elif y_prob >= 0.5:
        recommendations['immediate_actions'] = [
            'Discuss results with healthcare provider',
            'Baseline cognitive assessment',
            'Review family history'
        ]
    else:
        recommendations['immediate_actions'] = [
            'Share results at next routine medical visit',
            'Continue current preventive care'
        ]
    
    # Monitoring schedule
    if y_prob >= 0.7:
        recommendations['monitoring_schedule'] = {
            'cognitive_screening': 'Every 6 months',
            'brain_imaging': 'Annual or as recommended',
            'neuropsychological_testing': 'Every 12 months'
        }
    elif y_prob >= 0.5:
        recommendations['monitoring_schedule'] = {
            'cognitive_screening': 'Every 12 months',
            'specialized_testing': 'If symptoms develop'
        }
    else:
        recommendations['monitoring_schedule'] = {
            'routine_screening': 'Every 2-3 years',
            'symptom_monitoring': 'Ongoing self-assessment'
        }
    
    # Lifestyle recommendations
    recommendations['lifestyle_factors'] = [
        'Regular physical exercise (150 min/week moderate activity)',
        'Mediterranean-style diet rich in omega-3 fatty acids',
        'Adequate sleep (7-9 hours nightly)',
        'Social engagement and cognitive stimulation',
        'Stress management techniques',
        'Avoid smoking and limit alcohol consumption'
    ]
    
    # Clinical considerations
    if y_prob >= 0.6:
        recommendations['clinical_considerations'] = [
            'Genetic counseling if family history positive',
            'Cardiovascular risk factor management',
            'Depression and anxiety screening',
            'Medication review for cognitive effects'
        ]
    
    return recommendations

def format_explanation_text(explanation):
    """Format explanation as human-readable text."""
    
    text = f"""
{'='*60}
INDIVIDUAL PREDICTION EXPLANATION
{'='*60}

PARTICIPANT: {explanation['participant_info']['participant_id']}

PREDICTION RESULTS:
• Actual Outcome: {explanation['participant_info']['actual_outcome']}
• Predicted Outcome: {explanation['participant_info']['predicted_outcome']}
• Prediction Accuracy: {'[CORRECT]' if explanation['participant_info']['prediction_correct'] else '[INCORRECT]'}
• Risk Probability: {explanation['model_output']['risk_probability']:.1%}
• Risk Category: {explanation['model_output']['risk_category']}
• Confidence Level: {explanation['model_output']['confidence_level']}

{explanation['explanation_text']}

CLINICAL INTERPRETATION:
Risk Level: {explanation['clinical_interpretation']['risk_assessment']['level']}
Description: {explanation['clinical_interpretation']['risk_assessment']['description']}
Urgency: {explanation['clinical_interpretation']['risk_assessment']['urgency']}

Clinical Significance: {explanation['clinical_interpretation']['clinical_significance']['outcome']}
Meaning: {explanation['clinical_interpretation']['clinical_significance']['meaning']}

IMMEDIATE RECOMMENDATIONS:
"""
    
    for action in explanation['recommendations']['immediate_actions']:
        text += f"• {action}\n"
    
    text += f"\nMONITORING SCHEDULE:\n"
    for schedule_type, frequency in explanation['recommendations']['monitoring_schedule'].items():
        text += f"• {schedule_type.replace('_', ' ').title()}: {frequency}\n"
    
    text += f"\nLIFESTYLE RECOMMENDATIONS:\n"
    for lifestyle in explanation['recommendations']['lifestyle_factors']:
        text += f"• {lifestyle}\n"
    
    if explanation['recommendations']['clinical_considerations']:
        text += f"\nCLINICAL CONSIDERATIONS:\n"
        for consideration in explanation['recommendations']['clinical_considerations']:
            text += f"• {consideration}\n"
    
    text += f"\n{'='*60}\n"
    text += f"This explanation was generated by an AI system trained on brain imaging\n"
    text += f"and clinical data. Always consult with healthcare professionals for\n"
    text += f"medical decisions and interpretation of results.\n"
    text += f"{'='*60}\n"
    
    return text

def main():
    """Generate individual explanations for all participants."""
    
    base_path = Path(__file__).parent
    results_dir = base_path / 'results'
    individual_explanations_dir = results_dir / 'individual_explanations'
    individual_explanations_dir.mkdir(exist_ok=True, parents=True)
    
    predictions_file = results_dir / 'test_predictions_optimized.csv'
    
    if not predictions_file.exists():
        print(f"Error: {predictions_file} not found.")
        print("Please run the threshold optimization script first.")
        return
    
    # Load all participants
    with open(predictions_file, 'r') as f:
        lines = f.readlines()
    
    participant_ids = []
    for line in lines[1:]:  # Skip header
        parts = line.strip().split(',')
        if len(parts) >= 5:
            participant_ids.append(parts[4])
    
    print(f"Generating individual explanations for {len(participant_ids)} participants...")
    
    # Generate explanation for each participant
    all_explanations = {}
    for participant_id in participant_ids:
        explanation = explain_individual_prediction(
            participant_id, 
            predictions_file, 
            individual_explanations_dir
        )
        if explanation:
            all_explanations[participant_id] = explanation
    
    # Save summary of all explanations
    summary = {
        'total_participants': len(participant_ids),
        'explanations_generated': len(all_explanations),
        'summary_statistics': {
            'correct_predictions': sum(1 for exp in all_explanations.values() 
                                     if exp['participant_info']['prediction_correct']),
            'high_confidence_predictions': sum(1 for exp in all_explanations.values() 
                                             if exp['model_output']['confidence_level'] == 'High'),
            'high_risk_predictions': sum(1 for exp in all_explanations.values() 
                                       if exp['model_output']['risk_probability'] >= 0.7)
        }
    }
    
    with open(individual_explanations_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Individual explanations completed!")
    print(f"   Files saved to: {individual_explanations_dir}")
    print(f"   Explanations generated: {len(all_explanations)}/{len(participant_ids)}")
    print(f"   Correct predictions: {summary['summary_statistics']['correct_predictions']}/{len(all_explanations)}")

if __name__ == "__main__":
    main()