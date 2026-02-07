"""
Simple BiomedGPT-style Explainer (No Dependencies)
================================================
Generates rich explanations without complex dependencies.
"""

import json
import csv
from typing import Dict, Any
from datetime import datetime

class SimpleExplainer:
    """Simple explainer that generates detailed explanations."""
    
    def generate_explanation(self, patient_data: Dict, model_output: Dict) -> Dict[str, Any]:
        """Generate a comprehensive explanation."""
        
        # Safe data extraction
        participant_id = patient_data.get('participant_id', 'Unknown')
        age = self._safe_float(patient_data.get('age'), 60)
        sex = 'Female' if str(patient_data.get('sex', '')).strip() == '0' else 'Male'
        education = self._safe_float(patient_data.get('education'), 3)
        
        # Cognitive scores
        cvlt_7 = self._safe_float(patient_data.get('CVLT_7'), 13.5)
        rpm = self._safe_float(patient_data.get('RPM'), 50)
        bdi = self._safe_float(patient_data.get('BDI'), 5)
        dementia_history = self._safe_float(patient_data.get('dementia_history_parents'), 0)
        
        # Model outputs
        predicted_risk = model_output.get('predicted_risk', 'Unknown')
        confidence = model_output.get('confidence', 0.0)
        actual_label = model_output.get('actual_label', 'Unknown')
        agreement = model_output.get('agreement', False)
        
        # Clinical analysis
        clinical_analysis = self._analyze_clinical_data(age, cvlt_7, rpm, bdi, dementia_history)
        
        # Generate explanations
        doctor_explanation = self._generate_doctor_explanation(
            participant_id, age, sex, education, cvlt_7, rpm, bdi, dementia_history,
            predicted_risk, confidence, actual_label, agreement, clinical_analysis
        )
        
        patient_explanation = self._generate_patient_explanation(
            age, sex, cvlt_7, rpm, bdi, dementia_history,
            predicted_risk, confidence, clinical_analysis
        )
        
        return {
            'participant_id': participant_id,
            'clinical_summary': {
                'age': age, 'sex': sex, 'education': education,
                'cvlt_7': cvlt_7, 'rpm': rpm, 'bdi': bdi,
                'dementia_history': dementia_history
            },
            'model_prediction': model_output,
            'clinical_analysis': clinical_analysis,
            'doctor_explanation': doctor_explanation,
            'patient_explanation': patient_explanation,
            'timestamp': datetime.now().isoformat()
        }
    
    def _safe_float(self, value, default):
        """Safely convert to float with default."""
        try:
            return float(value or default)
        except (ValueError, TypeError):
            return default
    
    def _analyze_clinical_data(self, age, cvlt_7, rpm, bdi, dementia_history):
        """Analyze clinical data and extract insights."""
        
        # Memory assessment
        if cvlt_7 < 13.5:
            memory_status = "impaired"
            memory_concern = "significant"
        elif cvlt_7 >= 15:
            memory_status = "excellent"
            memory_concern = "none"
        else:
            memory_status = "normal"
            memory_concern = "minimal"
        
        # Cognitive reserve
        if rpm > 55:
            cognitive_reserve = "high"
        elif rpm < 45:
            cognitive_reserve = "low"
        else:
            cognitive_reserve = "moderate"
        
        # Mood assessment
        if bdi > 15:
            mood_status = "concerning"
        elif bdi < 5:
            mood_status = "excellent"
        else:
            mood_status = "normal"
        
        # Risk factors
        risk_factors = []
        protective_factors = []
        
        if memory_status == "impaired":
            risk_factors.append(f"Memory impairment (CVLT-7: {cvlt_7})")
        elif memory_status == "excellent":
            protective_factors.append(f"Excellent memory (CVLT-7: {cvlt_7})")
        
        if dementia_history > 0:
            risk_factors.append("Family history of dementia")
        else:
            protective_factors.append("No family history of dementia")
        
        if cognitive_reserve == "high":
            protective_factors.append(f"High cognitive reserve (RPM: {rpm})")
        elif cognitive_reserve == "low":
            risk_factors.append(f"Low cognitive reserve (RPM: {rpm})")
        
        if mood_status == "concerning":
            risk_factors.append(f"Depression concerns (BDI: {bdi})")
        elif mood_status == "excellent":
            protective_factors.append(f"Excellent mood (BDI: {bdi})")
        
        if age > 65:
            risk_factors.append(f"Advanced age ({age} years)")
        elif age < 55:
            protective_factors.append(f"Younger age ({age} years)")
        
        return {
            'memory_status': memory_status,
            'memory_concern': memory_concern,
            'cognitive_reserve': cognitive_reserve,
            'mood_status': mood_status,
            'risk_factors': risk_factors,
            'protective_factors': protective_factors,
            'risk_count': len(risk_factors),
            'protective_count': len(protective_factors)
        }
    
    def _generate_doctor_explanation(self, participant_id, age, sex, education, cvlt_7, rpm, bdi, 
                                   dementia_history, predicted_risk, confidence, actual_label, 
                                   agreement, clinical_analysis):
        """Generate explanation for healthcare providers."""
        
        risk_factors = clinical_analysis['risk_factors']
        protective_factors = clinical_analysis['protective_factors']
        
        summary = f"CLINICAL ASSESSMENT REPORT - {participant_id}"
        summary += f"\\nPatient: {age}-year-old {sex}, Education: {education} years"
        summary += f"\\nAI Prediction: {predicted_risk} (Confidence: {confidence:.1%})"
        summary += f"\\nDataset Label: {actual_label}"
        summary += f"\\nAgreement: {'Concordant' if agreement else 'Discordant'}"
        
        summary += "\\n\\nCOGNITIVE ASSESSMENT:"
        summary += f"\\n- Memory (CVLT-7): {cvlt_7} - {clinical_analysis['memory_status']}"
        summary += f"\\n- Fluid Intelligence (RPM): {rpm} - {clinical_analysis['cognitive_reserve']} cognitive reserve"
        summary += f"\\n- Mood (BDI): {bdi} - {clinical_analysis['mood_status']}"
        summary += f"\\n- Family History: {'Positive' if dementia_history > 0 else 'Negative'}"
        
        summary += "\\n\\nRISK ANALYSIS:"
        if risk_factors:
            summary += f"\\nRisk Factors ({len(risk_factors)}): " + "; ".join(risk_factors)
        else:
            summary += "\\nRisk Factors: None identified"
        
        if protective_factors:
            summary += f"\\nProtective Factors ({len(protective_factors)}): " + "; ".join(protective_factors)
        else:
            summary += "\\nProtective Factors: None identified"
        
        summary += "\\n\\nCLINICAL RECOMMENDATIONS:"
        recommendations = self._generate_clinical_recommendations(predicted_risk, confidence, clinical_analysis)
        for i, rec in enumerate(recommendations, 1):
            summary += f"\\n{i}. {rec}"
        
        return {
            'summary': summary,
            'risk_stratification': predicted_risk,
            'confidence_level': 'High' if confidence > 0.85 else 'Moderate' if confidence > 0.75 else 'Lower',
            'clinical_priority': 'High' if len(risk_factors) >= 3 else 'Moderate' if len(risk_factors) >= 1 else 'Routine',
            'recommendations': recommendations
        }
    
    def _generate_patient_explanation(self, age, sex, cvlt_7, rpm, bdi, dementia_history, 
                                    predicted_risk, confidence, clinical_analysis):
        """Generate patient-friendly explanation."""
        
        # Simple interpretations
        memory_simple = "some concerns" if cvlt_7 < 13.5 else "very good" if cvlt_7 >= 15 else "normal"
        thinking_simple = "strong" if rpm > 55 else "may need support" if rpm < 45 else "normal"
        mood_simple = "concerning" if bdi > 15 else "good"
        
        explanation = f"BRAIN HEALTH ASSESSMENT RESULTS\\n"
        explanation += f"Hello! We have completed your brain health evaluation using AI technology.\\n\\n"
        
        explanation += f"ABOUT YOU:\\n"
        explanation += f"You are a {age}-year-old {sex.lower()}.\\n\\n"
        
        explanation += f"YOUR TEST RESULTS:\\n"
        explanation += f"Memory Test: Your memory performance is {memory_simple} (score: {cvlt_7})\\n"
        explanation += f"Problem-Solving: Your thinking skills are {thinking_simple} (score: {rpm})\\n"
        explanation += f"Mood Assessment: {mood_simple} levels (score: {bdi})\\n"
        explanation += f"Family History: {'Yes' if dementia_history > 0 else 'No'} history of memory problems in parents\\n\\n"
        
        explanation += f"AI ASSESSMENT:\\n"
        explanation += f"The AI suggests {predicted_risk.lower().replace('risk', 'risk for cognitive changes')}. "
        explanation += f"The AI is {confidence:.0%} confident in this assessment.\\n\\n"
        
        explanation += f"WHAT THIS MEANS:\\n"
        if predicted_risk == "HIGH RISK":
            explanation += "This means we want to keep a closer eye on your brain health. "
            explanation += "This doesn't mean you will develop problems - many people with similar "
            explanation += "profiles maintain excellent brain health with proper care.\\n\\n"
        else:
            explanation += "This is encouraging news! The AI found patterns that are associated "
            explanation += "with better brain health. Keep up the good work with healthy choices.\\n\\n"
        
        explanation += f"RECOMMENDATIONS:\\n"
        patient_recs = self._generate_patient_recommendations(predicted_risk, clinical_analysis)
        for i, rec in enumerate(patient_recs, 1):
            explanation += f"{i}. {rec}\\n"
        
        return {
            'summary': explanation,
            'key_findings': {
                'memory': memory_simple,
                'thinking': thinking_simple,
                'mood': mood_simple,
                'family_history': 'yes' if dementia_history > 0 else 'no'
            },
            'ai_result': f"{predicted_risk} ({confidence:.0%} confidence)",
            'next_steps': patient_recs
        }
    
    def _generate_clinical_recommendations(self, predicted_risk, confidence, clinical_analysis):
        """Generate clinical recommendations."""
        recommendations = []
        
        if predicted_risk == "HIGH RISK":
            if confidence > 0.85:
                recommendations.append("Comprehensive neuropsychological evaluation within 3-6 months")
                if clinical_analysis['memory_status'] == 'impaired':
                    recommendations.append("Consider biomarker assessment if clinically indicated")
            else:
                recommendations.append("Enhanced cognitive monitoring with repeat assessment in 6-12 months")
            
            if clinical_analysis['mood_status'] == 'concerning':
                recommendations.append("Psychiatric evaluation for mood disorder management")
            
            if clinical_analysis['memory_status'] == 'impaired':
                recommendations.append("Memory training and cognitive rehabilitation referral")
            
            recommendations.append("Lifestyle interventions: Mediterranean diet, regular exercise, cognitive engagement")
        else:
            recommendations.append("Continue current cognitive health practices")
            recommendations.append("Annual cognitive screening with validated instruments")
            if clinical_analysis['protective_count'] > 0:
                recommendations.append("Maintain existing protective lifestyle factors")
        
        return recommendations
    
    def _generate_patient_recommendations(self, predicted_risk, clinical_analysis):
        """Generate patient-friendly recommendations."""
        recommendations = []
        
        if predicted_risk == "HIGH RISK":
            recommendations.append("Schedule a follow-up with your doctor to discuss these results")
            recommendations.append("Stay physically active with regular exercise")
            recommendations.append("Keep your mind active with reading, puzzles, or learning new skills")
            recommendations.append("Eat a healthy diet rich in fruits, vegetables, and omega-3 fatty acids")
            
            if clinical_analysis['mood_status'] == 'concerning':
                recommendations.append("Talk to your doctor about your mood and any concerns")
        else:
            recommendations.append("Continue your healthy lifestyle habits")
            recommendations.append("Keep up with regular medical check-ups")
            recommendations.append("Stay mentally and physically active")
            recommendations.append("Continue to challenge yourself with new activities")
        
        return recommendations

def run_simple_explanations():
    """Run the simple explanation pipeline."""
    
    print("Simple BiomedGPT-style Explainer")
    print("=" * 40)
    
    explainer = SimpleExplainer()
    
    # Load participant data
    participants = []
    try:
        with open("participants_with_labels.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            participants = list(reader)
        print(f"Loaded {len(participants)} participants")
    except Exception as e:
        print(f"Error loading participants: {e}")
        return
    
    # Load model results
    try:
        with open("results/real_dataset_explanation_summary.json", 'r') as f:
            model_results = json.load(f)
        print(f"Loaded model results")
    except Exception as e:
        print(f"Error loading model results: {e}")
        return
    
    # Process first few participants
    results = {'explanations': []}
    
    for result in model_results.get('test_results', [])[:3]:  # Process first 3
        participant_id = result['participant_id']
        
        # Find participant data
        participant_data = None
        for p in participants:
            if p.get('participant_id') == participant_id:
                participant_data = p
                break
        
        if not participant_data:
            continue
        
        print(f"Processing {participant_id}...")
        
        # Generate explanation
        explanation = explainer.generate_explanation(participant_data, result)
        results['explanations'].append(explanation)
    
    # Save results
    output_file = "results/simple_explanations.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Generated explanations for {len(results['explanations'])} participants")
    print(f"Results saved to: {output_file}")
    
    # Show sample
    if results['explanations']:
        sample = results['explanations'][0]
        print(f"\\nSample Patient Explanation for {sample['participant_id']}:")
        print("-" * 50)
        print(sample['patient_explanation']['summary'][:400] + "...")
        
        print(f"\\nModel Prediction: {sample['model_prediction']['predicted_risk']}")
        print(f"Confidence: {sample['model_prediction']['confidence']:.1%}")

if __name__ == "__main__":
    run_simple_explanations()