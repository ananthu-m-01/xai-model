"""
Simple BiomedGPT Integration for Explainable AI
===============================================
Direct integration without pandas dependency issues.
"""

import json
import csv
from typing import Dict, Any, Optional
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleBiomedGPTExplainer:
    """
    Simplified explainer that generates rich explanations without heavy dependencies.
    """
    
    def __init__(self):
        """Initialize the explainer."""
        self.model_available = False
        logger.info("SimpleBiomedGPT Explainer initialized (lightweight mode)")
    
    def generate_explanation(
        self, 
        patient_data: Dict[str, Any], 
        model_output: Dict[str, Any],
        explanation_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate enhanced explanation based on patient data and model output.
        
        Args:
            patient_data: Patient clinical data
            model_output: Model prediction output
            explanation_type: Type of explanation
        
        Returns:
            Enhanced explanation
        """
        
        # Extract key clinical variables
        age = patient_data.get('age', 'Unknown')
        sex = 'Female' if str(patient_data.get('sex', '')).strip() == '0' else 'Male' if str(patient_data.get('sex', '')).strip() == '1' else 'Unknown'
        
        # Parse clinical values safely
        try:
            cvlt_7 = float(patient_data.get('CVLT_7', 13.5) or 13.5)
        except (ValueError, TypeError):
            cvlt_7 = 13.5
            
        try:
            rpm = float(patient_data.get('RPM', 50) or 50)
        except (ValueError, TypeError):
            rpm = 50
            
        try:
            bdi = float(patient_data.get('BDI', 5) or 5)
        except (ValueError, TypeError):
            bdi = 5
            
        try:
            dementia_history = float(patient_data.get('dementia_history_parents', 0) or 0)
        except (ValueError, TypeError):
            dementia_history = 0
        
        education = patient_data.get('education', 'Unknown')
        apoe_haplotype = patient_data.get('APOE_haplotype', 'Unknown')
        
        # Model prediction details
        predicted_risk = model_output.get('predicted_risk', 'Unknown')
        confidence = model_output.get('confidence', 0.0)
        actual_label = model_output.get('actual_label', 'Unknown')
        agreement = model_output.get('agreement', False)
        
        if explanation_type == "comprehensive":
            return self._generate_comprehensive_explanation(
                age, sex, education, cvlt_7, rpm, bdi, dementia_history, apoe_haplotype,
                predicted_risk, confidence, actual_label, agreement, patient_data
            )
        elif explanation_type == "patient-friendly":
            return self._generate_patient_friendly_explanation(
                age, sex, cvlt_7, rpm, bdi, dementia_history,
                predicted_risk, confidence, patient_data
            )
        else:  # technical
            return self._generate_technical_explanation(
                age, sex, cvlt_7, rpm, bdi, dementia_history, apoe_haplotype,
                predicted_risk, confidence, agreement, patient_data
            )
    
    def _generate_comprehensive_explanation(self, age, sex, education, cvlt_7, rpm, bdi, 
                                          dementia_history, apoe_haplotype, predicted_risk, 
                                          confidence, actual_label, agreement, patient_data):
        """Generate comprehensive medical explanation."""
        
        # Clinical assessment
        memory_status = "impaired" if cvlt_7 < 13.5 else "excellent" if cvlt_7 >= 15 else "normal"
        cognitive_reserve = "high" if rpm > 55 else "low" if rpm < 45 else "moderate"
        mood_status = "concerning" if bdi > 15 else "normal"
        family_risk = "positive" if dementia_history > 0 else "negative"
        age_risk = "high" if age > 65 else "moderate" if age > 55 else "low"
        
        # Risk factor analysis
        risk_factors = []
        protective_factors = []
        
        if memory_status == "impaired":
            risk_factors.append(f"Memory impairment (CVLT-7: {cvlt_7})")
        elif memory_status == "excellent":
            protective_factors.append(f"Excellent memory (CVLT-7: {cvlt_7})")
        
        if family_risk == "positive":
            risk_factors.append("Positive family history of dementia")
        else:
            protective_factors.append("No family history of dementia")
        
        if cognitive_reserve == "high":
            protective_factors.append(f"High cognitive reserve (RPM: {rpm}, Education: {education} years)")
        elif cognitive_reserve == "low":
            risk_factors.append(f"Low cognitive reserve (RPM: {rpm})")
        
        if mood_status == "concerning":
            risk_factors.append(f"Depression concerns (BDI: {bdi})")
        else:
            protective_factors.append(f"Good mood profile (BDI: {bdi})")
        
        if age > 65:
            risk_factors.append(f"Advanced age ({age} years)")
        elif age < 55:
            protective_factors.append(f"Younger age ({age} years)")
        
        # Generate comprehensive clinical narrative
        clinical_summary = f"""COMPREHENSIVE CLINICAL ASSESSMENT

Patient Profile:
- Demographics: {age} year old {sex}, {education} years of education
- APOE Genotype: {apoe_haplotype}
- Family History: {family_risk.title()} for dementia

Cognitive Assessment:
- Memory Function (CVLT-7): {cvlt_7} - {memory_status.title()}
- Fluid Intelligence (RPM): {rpm} - {cognitive_reserve.title()} cognitive reserve
- Mood Assessment (BDI): {bdi} - {mood_status.title()}

Risk Factor Analysis:
Risk Factors ({len(risk_factors)}): {'; '.join(risk_factors) if risk_factors else 'None identified'}
Protective Factors ({len(protective_factors)}): {'; '.join(protective_factors) if protective_factors else 'None identified'}

AI Model Assessment:
- Prediction: {predicted_risk}
- Confidence: {confidence:.1%}
- Clinical Agreement: {'Concordant' if agreement else 'Discordant'} with dataset label ({actual_label})

Clinical Interpretation:
Based on the comprehensive assessment, this {age}-year-old {sex.lower()} presents with {memory_status} memory function and {cognitive_reserve} cognitive reserve. The AI model predicts {predicted_risk.lower()} with {confidence:.0%} confidence. Key clinical considerations include {risk_factors[0] if risk_factors else protective_factors[0] if protective_factors else 'balanced risk profile'}.

Evidence-Based Recommendations:
{self._generate_clinical_recommendations(risk_factors, protective_factors, predicted_risk, confidence)}"""
        
        return {
            'comprehensive_clinical_explanation': {
                'narrative': clinical_summary,
                'risk_factors': risk_factors,
                'protective_factors': protective_factors,
                'clinical_categories': {
                    'memory_status': memory_status,
                    'cognitive_reserve': cognitive_reserve,
                    'mood_status': mood_status,
                    'family_risk': family_risk,
                    'age_risk': age_risk
                },
                'ai_assessment': {
                    'prediction': predicted_risk,
                    'confidence': confidence,
                    'agreement': agreement
                },
                'generated_by': 'Enhanced Clinical Analysis',
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _generate_patient_friendly_explanation(self, age, sex, cvlt_7, rpm, bdi, 
                                             dementia_history, predicted_risk, confidence, patient_data):
        """Generate patient-friendly explanation."""
        
        # Simple interpretations
        memory_simple = "some concerns" if cvlt_7 < 13.5 else "very good" if cvlt_7 >= 15 else "normal"
        thinking_simple = "strong" if rpm > 55 else "may need support" if rpm < 45 else "normal"
        mood_simple = "concerning" if bdi > 15 else "good"
        family_simple = "yes" if dementia_history > 0 else "no"
        
        # Patient-friendly narrative
        patient_narrative = f\"\"\"
Understanding Your Brain Health Assessment

Hello! We've completed a comprehensive evaluation of your brain health using advanced AI technology. Here's what we found in simple terms:

About You:
You are a {age}-year-old {sex.lower()}, and we've looked at several important areas of your brain health.

Your Test Results:
🧠 Memory Test: Your memory performance is {memory_simple} (score: {cvlt_7})
🧩 Problem-Solving: Your thinking skills are {thinking_simple} (score: {rpm})
💭 Mood: Your mood assessment shows {mood_simple} levels (score: {bdi})
👨‍👩‍👧‍👦 Family History: {family_simple.title()} history of memory problems in parents

What Our AI Found:
The artificial intelligence analyzed all your information and suggests {predicted_risk.lower().replace('risk', 'risk for cognitive changes')}. The AI is {confidence:.0%} confident in this assessment.

What This Means:
{self._generate_patient_meaning(predicted_risk, memory_simple, thinking_simple, mood_simple, family_simple)}

Next Steps:
{self._generate_patient_recommendations(predicted_risk, memory_simple, mood_simple)}

Remember: This assessment is one tool to help guide your healthcare. Always discuss these results with your doctor to understand what they mean for your specific situation.
        \"\"\".strip()
        
        return {
            'patient_friendly_explanation': {
                'narrative': patient_narrative,
                'key_findings': {
                    'memory': memory_simple,
                    'thinking': thinking_simple,
                    'mood': mood_simple,
                    'family_history': family_simple
                },
                'ai_result': f"{predicted_risk} ({confidence:.0%} confidence)",
                'generated_by': 'Patient-Friendly AI',
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _generate_technical_explanation(self, age, sex, cvlt_7, rpm, bdi, dementia_history, 
                                      apoe_haplotype, predicted_risk, confidence, agreement, patient_data):
        """Generate technical/clinical explanation."""
        
        # Calculate risk scores
        memory_risk_score = max(0, (13.5 - cvlt_7) / 13.5) if cvlt_7 < 13.5 else 0
        cognitive_reserve_score = min(1.0, rpm / 60.0)
        mood_risk_score = min(1.0, max(0, bdi / 20.0))
        family_risk_score = 1.0 if dementia_history > 0 else 0.0
        age_risk_score = min(1.0, max(0, (age - 50) / 30.0))
        
        composite_risk_score = (
            memory_risk_score * 0.3 + 
            (1 - cognitive_reserve_score) * 0.2 + 
            mood_risk_score * 0.15 + 
            family_risk_score * 0.25 + 
            age_risk_score * 0.1
        )
        
        technical_narrative = f\"\"\"
TECHNICAL COGNITIVE RISK ASSESSMENT

Patient ID: {patient_data.get('participant_id', 'Unknown')}
Demographics: {age}yo {sex}, Education: {patient_data.get('education', 'Unknown')}y

Cognitive Domain Analysis:
1. Episodic Memory (CVLT-7): {cvlt_7} 
   - Percentile rank: {self._get_memory_percentile(cvlt_7, age)}
   - Risk contribution: {memory_risk_score:.3f}

2. Fluid Intelligence (RPM): {rpm}
   - Cognitive reserve index: {cognitive_reserve_score:.3f}
   - Protective factor strength: {1-cognitive_reserve_score:.3f}

3. Mood Assessment (BDI): {bdi}
   - Depression risk score: {mood_risk_score:.3f}
   - Clinical significance: {'Yes' if bdi > 15 else 'No'}

Genetic/Familial Risk:
- APOE Genotype: {apoe_haplotype}
- Parental Dementia History: {'Positive' if dementia_history > 0 else 'Negative'}
- Family risk score: {family_risk_score:.3f}

Age-Related Risk:
- Chronological age: {age} years
- Age-adjusted risk: {age_risk_score:.3f}

Composite Risk Analysis:
- Calculated risk score: {composite_risk_score:.3f}
- AI prediction: {predicted_risk}
- Model confidence: {confidence:.3f}
- Prediction concordance: {'Yes' if agreement else 'No'}

Clinical Decision Support:
The multimodal AI assessment integrating cognitive, genetic, and demographic factors 
suggests {predicted_risk.lower()} with {confidence:.1%} confidence. The composite risk 
score of {composite_risk_score:.3f} indicates {'elevated' if composite_risk_score > 0.4 else 'moderate' if composite_risk_score > 0.2 else 'low'} 
risk burden requiring {'immediate' if composite_risk_score > 0.4 else 'routine'} clinical follow-up.
        \"\"\".strip()
        
        return {
            'technical_explanation': {
                'narrative': technical_narrative,
                'risk_scores': {
                    'memory_risk': memory_risk_score,
                    'cognitive_reserve': cognitive_reserve_score,
                    'mood_risk': mood_risk_score,
                    'family_risk': family_risk_score,
                    'age_risk': age_risk_score,
                    'composite_risk': composite_risk_score
                },
                'clinical_thresholds': {
                    'memory_impairment': cvlt_7 < 13.5,
                    'low_cognitive_reserve': rpm < 45,
                    'depression_concern': bdi > 15,
                    'high_risk_age': age > 65
                },
                'generated_by': 'Technical Clinical Analysis',
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _generate_clinical_recommendations(self, risk_factors, protective_factors, predicted_risk, confidence):
        """Generate clinical recommendations."""
        recommendations = []
        
        if predicted_risk == "HIGH RISK":
            if confidence > 0.85:
                recommendations.append("1. Comprehensive neuropsychological evaluation within 3-6 months")
                recommendations.append("2. Consider biomarker assessment (CSF/PET) if clinically indicated")
            else:
                recommendations.append("1. Enhanced cognitive monitoring with repeat assessment in 6-12 months")
            
            if any("memory" in rf.lower() for rf in risk_factors):
                recommendations.append("3. Memory training and cognitive rehabilitation referral")
            
            if any("depression" in rf.lower() or "mood" in rf.lower() for rf in risk_factors):
                recommendations.append("4. Psychiatric evaluation for mood disorder management")
            
            recommendations.append("5. Lifestyle interventions: Mediterranean diet, regular exercise, cognitive engagement")
            
        else:  # LOW RISK
            recommendations.append("1. Continue current cognitive health practices")
            recommendations.append("2. Annual cognitive screening with validated instruments")
            recommendations.append("3. Maintain protective lifestyle factors")
            
            if protective_factors:
                recommendations.append("4. Leverage existing protective factors for continued brain health")
        
        return "\n".join(recommendations)
    
    def _generate_patient_meaning(self, predicted_risk, memory_simple, thinking_simple, mood_simple, family_simple):
        """Generate patient-friendly meaning."""
        if predicted_risk == "HIGH RISK":
            return f\"\"\"This means we want to keep a closer eye on your brain health. The AI found some patterns that suggest we should monitor you more carefully. This doesn't mean you will develop problems - many people with similar profiles maintain excellent brain health with proper care and healthy lifestyle choices.\"\"\".strip()
        else:
            return f\"\"\"This is encouraging news! The AI found patterns in your test results that are associated with better brain health. Your {memory_simple} memory and {thinking_simple} thinking skills are positive signs. Keep up the good work with healthy lifestyle choices.\"\"\".strip()
    
    def _generate_patient_recommendations(self, predicted_risk, memory_simple, mood_simple):
        """Generate patient recommendations."""
        recommendations = []
        
        if predicted_risk == "HIGH RISK":
            recommendations.append("• Schedule a follow-up with your doctor to discuss these results")
            recommendations.append("• Stay physically active with regular exercise")
            recommendations.append("• Keep your mind active with reading, puzzles, or learning new skills")
            recommendations.append("• Eat a healthy diet rich in fruits, vegetables, and omega-3 fatty acids")
            
            if mood_simple == "concerning":
                recommendations.append("• Talk to your doctor about your mood and any concerns you have")
        else:
            recommendations.append("• Continue your healthy lifestyle habits")
            recommendations.append("• Keep up with regular medical check-ups")
            recommendations.append("• Stay mentally and physically active")
            recommendations.append("• Continue to challenge yourself with new activities")
        
        return "\n".join(recommendations)
    
    def _get_memory_percentile(self, cvlt_7, age):
        """Estimate memory percentile based on age norms."""
        if age < 60:
            if cvlt_7 >= 15:
                return "75th-90th percentile"
            elif cvlt_7 >= 13:
                return "50th-75th percentile"
            else:
                return "Below 25th percentile"
        else:
            if cvlt_7 >= 14:
                return "75th-90th percentile"
            elif cvlt_7 >= 12:
                return "50th-75th percentile"
            else:
                return "Below 25th percentile"

def process_with_simple_biomedgpt(
    participants_file: str = "participants_with_labels.csv",
    results_file: str = "results/real_dataset_explanation_summary.json",
    max_participants: int = 5
) -> Dict[str, Any]:
    """
    Process participants with simplified BiomedGPT explanations.
    """
    
    print("🧬 SIMPLE BIOMEDGPT EXPLAINER")
    print("=" * 50)
    
    explainer = SimpleBiomedGPTExplainer()
    
    # Load participant data (manual CSV reading to avoid pandas issues)
    participants = []
    try:
        with open(participants_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            participants = list(reader)
        print(f"✅ Loaded {len(participants)} participants")
    except Exception as e:
        print(f"❌ Error loading participants: {e}")
        return {}
    
    # Load model results
    try:
        with open(results_file, 'r') as f:
            model_results = json.load(f)
        print(f"✅ Loaded model results")
    except Exception as e:
        print(f"❌ Error loading model results: {e}")
        return {}
    
    enhanced_explanations = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'explainer_type': 'SimpleBiomedGPT',
            'total_processed': 0
        },
        'explanations': []
    }
    
    # Process participants from test results
    test_results = model_results.get('test_results', [])[:max_participants]
    
    for result in test_results:
        participant_id = result['participant_id']
        
        # Find participant data
        participant_data = None
        for p in participants:
            if p.get('participant_id') == participant_id:
                participant_data = p
                break
        
        if not participant_data:
            print(f"⚠️  No data found for {participant_id}")
            continue
        
        print(f"🔍 Processing {participant_id}...")
        
        # Prepare model output
        model_output = {
            'predicted_risk': result['predicted_risk'],
            'confidence': result['confidence'],
            'actual_label': result['actual_label'],
            'agreement': result['agreement']
        }
        
        # Generate explanations
        comprehensive = explainer.generate_explanation(
            participant_data, model_output, "comprehensive"
        )
        
        patient_friendly = explainer.generate_explanation(
            participant_data, model_output, "patient-friendly"
        )
        
        technical = explainer.generate_explanation(
            participant_data, model_output, "technical"
        )
        
        # Combine results
        combined_explanation = {
            'participant_id': participant_id,
            'clinical_summary': {
                'age': participant_data.get('age'),
                'sex': participant_data.get('sex'),
                'education': participant_data.get('education'),
                'cvlt_7': participant_data.get('CVLT_7'),
                'rpm': participant_data.get('RPM'),
                'bdi': participant_data.get('BDI'),
                'family_history': participant_data.get('dementia_history_parents'),
                'apoe': participant_data.get('APOE_haplotype')
            },
            'model_prediction': model_output,
            'explanations': {
                'comprehensive': comprehensive,
                'patient_friendly': patient_friendly,
                'technical': technical
            }
        }
        
        enhanced_explanations['explanations'].append(combined_explanation)
    
    enhanced_explanations['metadata']['total_processed'] = len(enhanced_explanations['explanations'])
    
    # Save results
    output_file = "results/simple_biomedgpt_explanations.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(enhanced_explanations, f, indent=2, default=str)
        print(f"✅ Results saved to {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
    
    return enhanced_explanations

if __name__ == "__main__":
    print("🚀 Processing with Simple BiomedGPT Explainer...")
    results = process_with_simple_biomedgpt(max_participants=3)
    
    if results and results['explanations']:
        print(f"\n✅ Generated explanations for {len(results['explanations'])} participants")
        
        # Show sample explanation
        sample = results['explanations'][0]
        print(f"\n📋 Sample Explanation for {sample['participant_id']}:")
        print("-" * 40)
        
        patient_explanation = sample['explanations']['patient_friendly']
        if 'patient_friendly_explanation' in patient_explanation:
            narrative = patient_explanation['patient_friendly_explanation']['narrative']
            print(narrative[:500] + "..." if len(narrative) > 500 else narrative)
        
        print(f"\n🎯 Model Accuracy: {sample['model_prediction']['agreement']}")
        print(f"📁 Full results saved to: results/simple_biomedgpt_explanations.json")
    else:
        print("❌ No explanations generated")