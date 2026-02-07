"""
BiomedGPT Enhanced Explainer
============================
Integrates BiomedGPT to generate rich, explainable content from model outputs and patient data.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiomedGPTExplainer:
    """
    Enhanced explainer using BiomedGPT to generate rich explanations from model outputs and patient data.
    """
    
    def __init__(self, model_name: str = "PharMolix/BioMedGPT-LM-7B"):
        """
        Initialize BiomedGPT explainer.
        
        Args:
            model_name: HuggingFace model name for BiomedGPT
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try to load BiomedGPT
        self._load_biomedgpt()
    
    def _load_biomedgpt(self) -> bool:
        """Load BiomedGPT model and tokenizer."""
        try:
            logger.info(f"Loading BiomedGPT: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map="auto" if self.device.type == 'cuda' else None
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("✅ BiomedGPT loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load BiomedGPT: {e}")
            logger.info("Will use fallback explanation generation")
            return False
    
    def generate_biomedgpt_explanation(
        self, 
        patient_data: Dict[str, Any], 
        model_output: Dict[str, Any],
        explanation_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate enhanced explanation using BiomedGPT.
        
        Args:
            patient_data: Patient clinical data
            model_output: Model prediction output
            explanation_type: Type of explanation (comprehensive, technical, patient-friendly)
        
        Returns:
            Enhanced explanation with BiomedGPT-generated content
        """
        # Prepare clinical context for BiomedGPT
        clinical_prompt = self._create_clinical_prompt(patient_data, model_output, explanation_type)
        
        if self.model and self.tokenizer:
            # Generate with BiomedGPT
            biomedgpt_response = self._query_biomedgpt(clinical_prompt)
            explanation = self._parse_biomedgpt_response(biomedgpt_response, patient_data, model_output)
        else:
            # Fallback to enhanced clinical reasoning
            explanation = self._generate_fallback_explanation(patient_data, model_output, explanation_type)
        
        return explanation
    
    def _create_clinical_prompt(self, patient_data: Dict, model_output: Dict, explanation_type: str) -> str:
        """Create a comprehensive clinical prompt for BiomedGPT."""
        
        # Extract key clinical variables
        age = patient_data.get('age', 'Unknown')
        sex = 'Female' if patient_data.get('sex') == '0' else 'Male' if patient_data.get('sex') == '1' else 'Unknown'
        
        # Cognitive assessments
        cvlt_7 = patient_data.get('CVLT_7', 'Not available')
        rpm = patient_data.get('RPM', 'Not available') 
        bdi = patient_data.get('BDI', 'Not available')
        
        # Risk factors
        dementia_history = patient_data.get('dementia_history_parents', '0')
        apoe_haplotype = patient_data.get('APOE_haplotype', 'Unknown')
        
        # Lab values (if available)
        cholesterol = patient_data.get('total_cholesterol', 'Not available')
        hdl = patient_data.get('cholesterol_HDL', 'Not available')
        
        # Model prediction
        predicted_risk = model_output.get('predicted_risk', 'Unknown')
        confidence = model_output.get('confidence', 0.0)
        
        if explanation_type == "comprehensive":
            prompt = f"""
You are a specialized medical AI assistant with expertise in cognitive assessment and dementia risk prediction. 

PATIENT CLINICAL PROFILE:
- Patient ID: {patient_data.get('participant_id', 'Unknown')}
- Demographics: {age} year old {sex}
- Education Level: {patient_data.get('education', 'Unknown')} years
- Family History: {'Positive' if float(dementia_history or 0) > 0 else 'Negative'} for dementia in parents
- APOE Genotype: {apoe_haplotype}

COGNITIVE ASSESSMENT RESULTS:
- CVLT-7 (Memory): {cvlt_7}
- RPM (Fluid Intelligence): {rpm}
- BDI (Depression): {bdi}

LABORATORY VALUES:
- Total Cholesterol: {cholesterol}
- HDL Cholesterol: {hdl}

AI MODEL PREDICTION:
- Risk Assessment: {predicted_risk}
- Confidence Level: {confidence:.1%}

Please provide a comprehensive medical explanation that includes:
1. Clinical interpretation of the cognitive test results
2. Risk factor analysis considering family history and genetics
3. Laboratory findings relevance to cognitive health
4. AI model assessment accuracy and clinical implications
5. Evidence-based recommendations for patient management

Base your explanation on current medical literature and clinical guidelines for cognitive assessment and dementia prevention.
"""
        
        elif explanation_type == "technical":
            prompt = f"""
As a medical AI specialist, analyze this cognitive risk assessment case:

CASE DATA:
Patient: {age}yo {sex}, Education: {patient_data.get('education')}y
Cognitive: CVLT-7={cvlt_7}, RPM={rpm}, BDI={bdi}
Genetics: APOE={apoe_haplotype}, Family Hx={'Positive' if float(dementia_history or 0) > 0 else 'Negative'}
AI Prediction: {predicted_risk} ({confidence:.1%} confidence)

Provide technical analysis including:
1. Cognitive domain-specific interpretation
2. Genetic risk stratification
3. Model prediction accuracy assessment
4. Clinical decision support recommendations
"""
        
        else:  # patient-friendly
            prompt = f"""
Explain this brain health assessment in patient-friendly language:

Patient: {age} year old {sex}
Memory test score: {cvlt_7}
Problem-solving score: {rpm}
Mood assessment: {bdi}
Family history: {'Yes' if float(dementia_history or 0) > 0 else 'No'} dementia in parents
AI assessment: {predicted_risk} (confidence: {confidence:.0%})

Provide a clear, compassionate explanation that:
1. Explains what these tests measure
2. Interprets the results in understandable terms
3. Discusses the AI assessment
4. Offers practical next steps and recommendations
5. Addresses likely patient concerns

Use simple language and avoid medical jargon.
"""
        
        return prompt
    
    def _query_biomedgpt(self, prompt: str, max_length: int = 800) -> str:
        """Query BiomedGPT with the clinical prompt."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            if prompt in response:
                generated_text = response.replace(prompt, "").strip()
            else:
                generated_text = response
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error querying BiomedGPT: {e}")
            return "Error generating BiomedGPT response. Using fallback explanation."
    
    def _parse_biomedgpt_response(self, response: str, patient_data: Dict, model_output: Dict) -> Dict[str, Any]:
        """Parse and structure BiomedGPT response."""
        
        return {
            'biomedgpt_explanation': {
                'raw_response': response,
                'generated_by': 'BiomedGPT',
                'clinical_interpretation': self._extract_clinical_sections(response),
                'confidence_level': 'High' if len(response) > 200 else 'Medium',
                'timestamp': datetime.now().isoformat()
            },
            'enhanced_clinical_summary': {
                'patient_id': patient_data.get('participant_id'),
                'ai_assessment': model_output.get('predicted_risk'),
                'model_confidence': model_output.get('confidence'),
                'biomedgpt_insights': response[:500],  # First 500 chars as summary
                'explanation_quality': 'AI-Enhanced'
            }
        }
    
    def _extract_clinical_sections(self, response: str) -> Dict[str, str]:
        """Extract structured sections from BiomedGPT response."""
        sections = {}
        
        # Look for numbered sections or bullet points
        lines = response.split('\n')
        current_section = 'main'
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers (numbers, bullets, etc.)
            if any(line.startswith(marker) for marker in ['1.', '2.', '3.', '4.', '5.', '•', '-', '*']):
                # Save previous section
                if current_content:
                    sections[current_section] = ' '.join(current_content)
                    current_content = []
                
                # Start new section
                current_section = line[:20]  # Use first 20 chars as section key
                current_content = [line]
            else:
                current_content.append(line)
        
        # Save final section
        if current_content:
            sections[current_section] = ' '.join(current_content)
        
        return sections
    
    def _generate_fallback_explanation(self, patient_data: Dict, model_output: Dict, explanation_type: str) -> Dict[str, Any]:
        """Generate enhanced explanation without BiomedGPT (fallback)."""
        
        # Extract clinical data
        age = patient_data.get('age', 60)
        cvlt_7 = float(patient_data.get('CVLT_7', 13.5) or 13.5)
        rpm = float(patient_data.get('RPM', 50) or 50)
        bdi = float(patient_data.get('BDI', 5) or 5)
        dementia_history = float(patient_data.get('dementia_history_parents', 0) or 0)
        
        predicted_risk = model_output.get('predicted_risk', 'Unknown')
        confidence = model_output.get('confidence', 0.0)
        
        # Generate clinical reasoning
        clinical_reasoning = []
        
        # Memory assessment
        if cvlt_7 < 13.5:
            clinical_reasoning.append(f"Memory performance (CVLT-7: {cvlt_7}) indicates impairment requiring monitoring")
        elif cvlt_7 >= 15:
            clinical_reasoning.append(f"Memory performance (CVLT-7: {cvlt_7}) is excellent and protective")
        else:
            clinical_reasoning.append(f"Memory performance (CVLT-7: {cvlt_7}) is within normal range")
        
        # Cognitive reserve
        if rpm > 55:
            clinical_reasoning.append(f"Fluid intelligence (RPM: {rpm}) suggests good cognitive reserve")
        elif rpm < 45:
            clinical_reasoning.append(f"Fluid intelligence (RPM: {rpm}) may indicate cognitive vulnerability")
        
        # Mood factors
        if bdi > 15:
            clinical_reasoning.append(f"Depression scores (BDI: {bdi}) require clinical attention")
        elif bdi < 10:
            clinical_reasoning.append(f"Mood assessment (BDI: {bdi}) is within healthy range")
        
        # Family history
        if dementia_history > 0:
            clinical_reasoning.append("Positive family history increases genetic risk burden")
        else:
            clinical_reasoning.append("No reported family history of dementia")
        
        # AI assessment interpretation
        if confidence > 0.85:
            confidence_desc = "high confidence"
        elif confidence > 0.75:
            confidence_desc = "moderate confidence"
        else:
            confidence_desc = "lower confidence"
        
        clinical_reasoning.append(f"AI model predicts {predicted_risk.lower()} with {confidence_desc} ({confidence:.1%})")
        
        explanation_text = ". ".join(clinical_reasoning) + "."
        
        return {
            'enhanced_clinical_explanation': {
                'reasoning': explanation_text,
                'generated_by': 'Enhanced Clinical Logic',
                'clinical_factors': {
                    'memory_status': 'impaired' if cvlt_7 < 13.5 else 'excellent' if cvlt_7 >= 15 else 'normal',
                    'cognitive_reserve': 'high' if rpm > 55 else 'low' if rpm < 45 else 'moderate',
                    'mood_status': 'concerning' if bdi > 15 else 'normal',
                    'family_history': 'positive' if dementia_history > 0 else 'negative'
                },
                'ai_assessment': {
                    'prediction': predicted_risk,
                    'confidence': confidence,
                    'confidence_level': confidence_desc
                },
                'timestamp': datetime.now().isoformat()
            }
        }

def process_patient_cohort_with_biomedgpt(
    participants_file: str = "participants_with_labels.csv",
    results_file: str = "results/real_dataset_explanation_summary.json",
    output_file: str = "results/biomedgpt_enhanced_explanations.json"
) -> Dict[str, Any]:
    """
    Process the entire patient cohort with BiomedGPT explanations.
    
    Args:
        participants_file: Path to participant data CSV
        results_file: Path to model results JSON
        output_file: Path to save enhanced explanations
    
    Returns:
        Dictionary with enhanced explanations for all patients
    """
    # Initialize BiomedGPT explainer
    explainer = BiomedGPTExplainer()
    
    # Load participant data
    try:
        participants_df = pd.read_csv(participants_file)
        logger.info(f"✅ Loaded {len(participants_df)} participants")
    except Exception as e:
        logger.error(f"Error loading participants: {e}")
        return {}
    
    # Load model results
    try:
        with open(results_file, 'r') as f:
            model_results = json.load(f)
        logger.info(f"✅ Loaded model results")
    except Exception as e:
        logger.error(f"Error loading model results: {e}")
        return {}
    
    enhanced_explanations = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'explainer_model': explainer.model_name,
            'total_participants': len(model_results.get('test_results', [])),
            'biomedgpt_available': explainer.model is not None
        },
        'explanations': []
    }
    
    # Process each participant in test results
    for result in model_results.get('test_results', []):
        participant_id = result['participant_id']
        
        # Get participant clinical data
        participant_data = participants_df[
            participants_df['participant_id'] == participant_id
        ].iloc[0].to_dict() if len(participants_df[participants_df['participant_id'] == participant_id]) > 0 else {}
        
        if not participant_data:
            logger.warning(f"No clinical data found for {participant_id}")
            continue
        
        # Create model output dict
        model_output = {
            'predicted_risk': result['predicted_risk'],
            'confidence': result['confidence'],
            'actual_label': result['actual_label'],
            'agreement': result['agreement']
        }
        
        logger.info(f"Generating BiomedGPT explanation for {participant_id}...")
        
        # Generate comprehensive explanation
        comprehensive_explanation = explainer.generate_biomedgpt_explanation(
            patient_data=participant_data,
            model_output=model_output,
            explanation_type="comprehensive"
        )
        
        # Generate patient-friendly explanation
        patient_explanation = explainer.generate_biomedgpt_explanation(
            patient_data=participant_data,
            model_output=model_output,
            explanation_type="patient-friendly"
        )
        
        # Combine explanations
        enhanced_explanations['explanations'].append({
            'participant_id': participant_id,
            'clinical_data': {
                'age': participant_data.get('age'),
                'sex': participant_data.get('sex'),
                'cvlt_7': participant_data.get('CVLT_7'),
                'rpm': participant_data.get('RPM'),
                'bdi': participant_data.get('BDI'),
                'dementia_history': participant_data.get('dementia_history_parents'),
                'apoe_haplotype': participant_data.get('APOE_haplotype')
            },
            'model_prediction': model_output,
            'comprehensive_explanation': comprehensive_explanation,
            'patient_friendly_explanation': patient_explanation
        })
    
    # Save enhanced explanations
    try:
        with open(output_file, 'w') as f:
            json.dump(enhanced_explanations, f, indent=2, default=str)
        logger.info(f"✅ Enhanced explanations saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving explanations: {e}")
    
    return enhanced_explanations

if __name__ == "__main__":
    print("🧬 BiomedGPT Enhanced Explainer")
    print("=" * 60)
    
    # Process cohort with BiomedGPT
    results = process_patient_cohort_with_biomedgpt()
    
    if results:
        print(f"✅ Generated enhanced explanations for {len(results.get('explanations', []))} participants")
        print(f"🤖 BiomedGPT Available: {results['metadata']['biomedgpt_available']}")
        print("📄 Results saved to: results/biomedgpt_enhanced_explanations.json")
    else:
        print("❌ Failed to generate enhanced explanations")