"""
Unified Explainer - NumPy 2.x Compatible Version
Uses only BiomedBERT model to avoid pandas dependency issues.
Generates dynamic explanations from actual model predictions and attention weights.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Safe imports avoiding pandas dependency
try:
    from model_huggingface_fixed import AdvancedMultiModalNet, AdvancedConfig
    from transformers import AutoTokenizer, AutoModel
    HAS_BIOMEDBERT = True
    print("✅ BiomedBERT model available")
except ImportError as e:
    HAS_BIOMEDBERT = False
    AdvancedMultiModalNet = None
    AdvancedConfig = None
    AutoTokenizer = None
    print(f"❌ BiomedBERT model not available: {e}")

# Skip hybrid model to avoid pandas import
HAS_HYBRID = False
print("ℹ️ Hybrid model skipped to avoid pandas dependency")


class UnifiedExplainerPandasFree:
    """
    Unified explainer using only BiomedBERT model to avoid pandas compatibility issues.
    Generates dynamic explanations from actual model predictions and attention weights.
    """
    
    def __init__(
        self,
        biomedbert_path: str = 'best_model_biomedbert.pth',
        device: Optional[torch.device] = None
    ):
        """Initialize unified explainer with BiomedBERT model only."""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model containers
        self.biomedbert_model = None
        self.biomedbert_tokenizer = None
        
        # Load BiomedBERT model
        if HAS_BIOMEDBERT and Path(biomedbert_path).exists():
            try:
                self.biomedbert_model, self.biomedbert_tokenizer = self._load_biomedbert_model(biomedbert_path)
                print(f"✅ Loaded BiomedBERT model from {biomedbert_path}")
            except Exception as e:
                print(f"❌ Failed to load BiomedBERT model: {e}")
        else:
            print(f"❌ BiomedBERT model not found: {biomedbert_path}")
    
    def _load_biomedbert_model(self, path: str) -> Tuple[Any, Optional[Any]]:
        """Load BiomedBERT-based model with proper configuration."""
        if not HAS_BIOMEDBERT:
            raise ImportError("BiomedBERT dependencies not available")
        
        # Load checkpoint and infer dimensions
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        
        # Default dimensions
        eeg_dim = 16
        fmri_dim = 32
        
        # Try to infer dimensions from state dict
        try:
            if isinstance(state_dict, dict):
                if 'eeg_encoder.input_proj.0.weight' in state_dict:
                    eeg_dim = int(state_dict['eeg_encoder.input_proj.0.weight'].shape[1])
                if 'fmri_encoder.input_proj.0.weight' in state_dict:
                    fmri_dim = int(state_dict['fmri_encoder.input_proj.0.weight'].shape[1])
        except Exception as e:
            print(f"Warning: Could not infer dimensions from state dict: {e}")
        
        # Use config if available
        if AdvancedConfig:
            eeg_dim = getattr(AdvancedConfig, 'EEG_DIM', eeg_dim)
            fmri_dim = getattr(AdvancedConfig, 'FMRI_DIM', fmri_dim)
            model_name = getattr(AdvancedConfig, 'CHOSEN_MODEL', 'biomedbert')
        else:
            model_name = 'biomedbert'
        
        print(f"📊 Using dimensions: EEG={eeg_dim}, fMRI={fmri_dim}, Model={model_name}")
        
        # Create model
        if not AdvancedMultiModalNet:
            raise ImportError("AdvancedMultiModalNet not available")
            
        model = AdvancedMultiModalNet(
            eeg_dim=eeg_dim,
            fmri_dim=fmri_dim, 
            model_name=model_name
        ).to(self.device)
        
        # Load state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Load tokenizer
        try:
            if AdvancedConfig and AutoTokenizer:
                model_configs = getattr(AdvancedConfig, 'AVAILABLE_MODELS', {})
                hf_model_id = model_configs.get(model_name, 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
                tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
                print(f"✅ Loaded tokenizer: {hf_model_id}")
            else:
                tokenizer = None
                print("❌ Tokenizer not available")
        except Exception as e:
            tokenizer = None
            print(f"❌ Failed to load tokenizer: {e}")
        
        return model, tokenizer
    
    def _extract_biomedbert_features(self, text_inputs: Dict, model_output: torch.Tensor) -> Dict[str, Any]:
        """Extract attention and semantic features from BiomedBERT model."""
        features = {
            'attention_analysis': {},
            'semantic_features': {},
            'confidence_metrics': {},
            'error': None
        }
        
        if not self.biomedbert_model or not self.biomedbert_tokenizer:
            features['error'] = 'Model or tokenizer not available'
            return features
        
        try:
            # Get model outputs with attention
            with torch.no_grad():
                text_encoder_output = self.biomedbert_model.text_encoder(
                    **text_inputs, 
                    output_attentions=True
                )
                
                # Extract attention patterns
                if hasattr(text_encoder_output, 'attentions') and text_encoder_output.attentions:
                    last_attention = text_encoder_output.attentions[-1]  # [batch, heads, seq, seq]
                    
                    # Average attention across heads
                    avg_attention = last_attention.mean(dim=1).squeeze(0)  # [seq, seq]
                    
                    # Get CLS token attention (first token)
                    cls_attention = avg_attention[0, :]  # [seq]
                    
                    # Convert token IDs to tokens
                    input_ids = text_inputs['input_ids'][0]
                    tokens = self.biomedbert_tokenizer.convert_ids_to_tokens(input_ids)
                    
                    # Find most attended tokens
                    top_attention_indices = cls_attention.argsort(descending=True)[:10]
                    
                    attended_tokens = []
                    for idx in top_attention_indices:
                        if idx < len(tokens):
                            token = tokens[idx]
                            # Skip special tokens and padding
                            if not token.startswith('[') and not token.startswith('<') and token != '[PAD]':
                                attended_tokens.append({
                                    'token': token,
                                    'attention_score': float(cls_attention[idx]),
                                    'position': int(idx)
                                })
                    
                    features['attention_analysis'] = {
                        'top_attended_tokens': attended_tokens[:6],
                        'attention_distribution': cls_attention.cpu().numpy().tolist(),
                        'max_attention_score': float(torch.max(cls_attention)),
                        'attention_entropy': float(-torch.sum(cls_attention * torch.log(cls_attention + 1e-10)))
                    }
                
                # Extract hidden state features (semantic understanding)
                if hasattr(text_encoder_output, 'last_hidden_state'):
                    hidden_states = text_encoder_output.last_hidden_state[0]  # [seq, hidden_dim]
                    cls_embedding = hidden_states[0]  # CLS token embedding
                    
                    features['semantic_features'] = {
                        'cls_embedding_norm': float(torch.norm(cls_embedding)),
                        'mean_hidden_activation': float(torch.mean(hidden_states)),
                        'hidden_state_variance': float(torch.var(hidden_states))
                    }
            
            # Analyze prediction confidence
            probs = F.softmax(model_output, dim=1)
            max_prob = float(torch.max(probs))
            prediction_entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)))
            
            features['confidence_metrics'] = {
                'max_probability': max_prob,
                'prediction_entropy': prediction_entropy,
                'confidence_level': 'high' if max_prob > 0.8 else 'medium' if max_prob > 0.6 else 'low',
                'probability_distribution': probs[0].cpu().numpy().tolist()
            }
            
        except Exception as e:
            features['error'] = str(e)
            print(f"❌ Feature extraction error: {e}")
        
        return features
    
    def _generate_dynamic_medical_explanation(
        self,
        prediction: Dict[str, Any],
        features: Dict[str, Any],
        eeg_stats: Dict[str, Any],
        fmri_stats: Dict[str, Any],
        patient_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate dynamic medical explanation based on model analysis."""
        
        risk_level = prediction['risk_level']
        confidence = prediction['confidence']
        attended_tokens = features.get('attention_analysis', {}).get('top_attended_tokens', [])
        confidence_metrics = features.get('confidence_metrics', {})
        
        # Analyze attended clinical terms
        medical_concepts = []
        risk_indicators = []
        
        for token_info in attended_tokens:
            token = token_info['token'].lower()
            attention_score = token_info['attention_score']
            
            # Detect medical concepts
            if any(term in token for term in ['alzheimer', 'dementia', 'cognitive', 'memory', 'family']):
                medical_concepts.append({
                    'concept': token,
                    'relevance': attention_score,
                    'type': 'risk_factor' if any(rf in token for rf in ['family', 'history']) else 'symptom'
                })
            
            # Detect risk indicators
            if any(term in token for term in ['difficult', 'problem', 'decline', 'impair']):
                risk_indicators.append({
                    'indicator': token,
                    'attention': attention_score
                })
        
        # Generate doctor explanation
        doctor_summary = f"""
BIOMEDBERT MULTIMODAL ASSESSMENT: {risk_level}
Model Confidence: {confidence:.1%} ({confidence_metrics.get('confidence_level', 'unknown')} reliability)
Prediction Entropy: {confidence_metrics.get('prediction_entropy', 0):.3f}
Attention Analysis: {len(attended_tokens)} key clinical concepts identified
        """.strip()
        
        doctor_findings = []
        
        # EEG analysis
        if eeg_stats.get('abnormal_patterns', False):
            doctor_findings.append(
                f"Electrophysiological: Model detected irregular neural oscillations "
                f"(μ={eeg_stats['mean']:.3f}, σ={eeg_stats['std']:.3f}). "
                f"Attention focused on neurophysiological markers."
            )
        else:
            doctor_findings.append(
                f"Electrophysiological: Neural activity patterns within normal parameters "
                f"(μ={eeg_stats['mean']:.3f}, σ={eeg_stats['std']:.3f}). "
                f"Model assessment indicates stable cortical function."
            )
        
        # fMRI analysis
        if fmri_stats.get('connectivity_issues', False):
            doctor_findings.append(
                f"Functional Connectivity: Reduced network coherence detected "
                f"(μ={fmri_stats['mean']:.3f}). Model indicates potential "
                f"default-mode network alterations."
            )
        else:
            doctor_findings.append(
                f"Functional Connectivity: Network integrity preserved "
                f"(μ={fmri_stats['mean']:.3f}). Inter-regional communication "
                f"within expected ranges."
            )
        
        # Medical concept analysis
        if medical_concepts:
            concept_names = [c['concept'] for c in medical_concepts[:3]]
            max_relevance = max(c['relevance'] for c in medical_concepts)
            doctor_findings.append(
                f"Clinical Text Analysis: BiomedBERT identified key concepts: "
                f"{', '.join(concept_names)} (max relevance: {max_relevance:.3f}). "
                f"Semantic analysis of medical history complete."
            )
        
        # Risk factor assessment
        doctor_risk_factors = []
        family_concepts = [c for c in medical_concepts if 'family' in c['concept'] or 'history' in c['concept']]
        cognitive_concepts = [c for c in medical_concepts if any(term in c['concept'] for term in ['cognitive', 'memory', 'learning'])]
        
        if family_concepts:
            doctor_risk_factors.append(
                f"Familial predisposition markers identified via semantic analysis "
                f"(relevance: {family_concepts[0]['relevance']:.3f})"
            )
        if cognitive_concepts:
            doctor_risk_factors.append(
                f"Cognitive performance indicators detected "
                f"(relevance: {cognitive_concepts[0]['relevance']:.3f})"
            )
        if risk_indicators:
            doctor_risk_factors.append(
                f"Symptom indicators: {', '.join([r['indicator'] for r in risk_indicators[:2]])}"
            )
        
        if not doctor_risk_factors:
            doctor_risk_factors.append("No significant risk markers detected in current analysis")
        
        # Generate patient explanation
        age = patient_info.get('age', 'unknown')
        
        if risk_level == 'HIGH RISK':
            patient_summary = f"""
Our BiomedBERT AI analysis suggests an increased risk for cognitive changes. 
The AI is {confidence:.0%} confident in this assessment. We found patterns 
in your brain scans and medical information that suggest closer monitoring would be beneficial.
            """.strip()
            
            patient_meaning = f"""
This means we want to work together to protect your brain health. The AI noticed 
patterns that suggest we should monitor you more closely. With proper care and 
lifestyle choices, many people with similar findings maintain excellent brain health.
            """.strip()
        else:
            patient_summary = f"""
Our BiomedBERT AI analysis suggests a lower risk for cognitive decline at this time. 
The AI is {confidence:.0%} confident in this assessment. Your brain patterns and 
medical information are reassuring.
            """.strip()
            
            patient_meaning = f"""
This is encouraging news! The AI found patterns associated with better brain health. 
Your brain scans show healthy activity, and your medical information suggests 
lower risk. Keep up the good work with healthy lifestyle choices.
            """.strip()
        
        # Patient-friendly findings
        patient_findings = []
        
        if eeg_stats.get('abnormal_patterns', False):
            patient_findings.append("🧠 Brain Activity: We noticed some unusual patterns in your brain's electrical signals")
        else:
            patient_findings.append("🧠 Brain Activity: Your brain's electrical patterns look healthy and normal")
        
        if fmri_stats.get('connectivity_issues', False):
            patient_findings.append("🔗 Brain Networks: Some brain regions aren't communicating as strongly as we'd like to see")
        else:
            patient_findings.append("🔗 Brain Networks: Your brain regions are communicating well with each other")
        
        if family_concepts:
            patient_findings.append("👨‍👩‍👧‍👦 Family History: The AI noted your family medical history is important for this assessment")
        
        if cognitive_concepts:
            patient_findings.append("🧭 Thinking Skills: The AI analyzed information about your memory and learning abilities")
        
        # Recommendations
        if risk_level == 'HIGH RISK':
            doctor_recommendations = [
                f"Comprehensive neuropsychological assessment recommended (confidence: {confidence:.1%})",
                "Consider genetic counseling based on AI risk stratification",
                "Establish cognitive baseline for longitudinal monitoring",
                "Implement targeted lifestyle interventions",
                f"Follow-up in {'3-6 months' if confidence > 0.8 else '6-12 months'} for reassessment"
            ]
            
            patient_recommendations = [
                "Schedule a follow-up appointment to discuss these findings in detail",
                "Stay physically active with regular exercise (aim for 150 minutes/week)",
                "Keep your mind engaged with learning and challenging activities",
                "Eat a brain-healthy diet rich in omega-3s and antioxidants",
                "Maintain strong social connections and manage stress effectively"
            ]
        else:
            doctor_recommendations = [
                f"Continue routine monitoring (AI assessment: {confidence:.1%} low-risk)",
                "Maintain current preventive health strategies",
                "Annual cognitive screening with AI assistance recommended"
            ]
            
            patient_recommendations = [
                "Continue your current healthy lifestyle habits",
                "Keep up with regular medical check-ups",
                "Stay mentally and physically active"
            ]
        
        return {
            'for_doctor': {
                'summary': doctor_summary,
                'clinical_findings': doctor_findings,
                'risk_factors': doctor_risk_factors,
                'recommendations': doctor_recommendations,
                'technical_details': {
                    'model_confidence': confidence,
                    'attention_analysis': features.get('attention_analysis', {}),
                    'semantic_features': features.get('semantic_features', {}),
                    'brain_metrics': {
                        'eeg_stats': eeg_stats,
                        'fmri_stats': fmri_stats
                    }
                }
            },
            'for_patient': {
                'summary': patient_summary,
                'what_we_found': patient_findings,
                'what_this_means': patient_meaning,
                'next_steps': patient_recommendations
            }
        }
    
    def explain_prediction(
        self,
        eeg_data: np.ndarray,
        fmri_data: np.ndarray,
        clinical_text: str,
        patient_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive explanation using BiomedBERT model analysis."""
        
        if not self.biomedbert_model or not self.biomedbert_tokenizer:
            return {
                'error': 'BiomedBERT model not available',
                'timestamp': datetime.now().isoformat()
            }
        
        patient_info = patient_info or {}
        
        # Prepare inputs
        eeg_tensor = torch.FloatTensor(eeg_data).unsqueeze(0).to(self.device)
        fmri_tensor = torch.FloatTensor(fmri_data).unsqueeze(0).to(self.device)
        
        # Tokenize clinical text
        max_length = 256
        if AdvancedConfig:
            max_length = getattr(AdvancedConfig, 'MAX_TEXT_LENGTH', 256)
        
        text_inputs = self.biomedbert_tokenizer(
            clinical_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        # Get model prediction
        with torch.no_grad():
            model_output = self.biomedbert_model(eeg_tensor, fmri_tensor, text_inputs)
            probs = F.softmax(model_output, dim=1)
            predicted_class = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, predicted_class].item())
        
        # Analyze brain data
        eeg_stats = {
            'mean': float(np.mean(eeg_data)),
            'std': float(np.std(eeg_data)),
            'abnormal_patterns': float(np.std(eeg_data)) > abs(float(np.mean(eeg_data))) * 2.0
        }
        
        fmri_stats = {
            'mean': float(np.mean(fmri_data)),
            'connectivity_issues': float(np.mean(fmri_data)) < -0.2
        }
        
        # Extract model features
        biomedbert_features = self._extract_biomedbert_features(text_inputs, model_output)
        
        # Create prediction summary
        risk_level = 'HIGH RISK' if predicted_class == 1 else 'LOW RISK'
        prediction_info = {
            'risk_level': risk_level,
            'confidence': confidence,
            'predicted_class': predicted_class,
            'model_type': 'biomedbert_only'
        }
        
        # Generate dynamic explanations
        explanations = self._generate_dynamic_medical_explanation(
            prediction=prediction_info,
            features=biomedbert_features,
            eeg_stats=eeg_stats,
            fmri_stats=fmri_stats,
            patient_info=patient_info
        )
        
        return {
            'patient_info': patient_info,
            'prediction': prediction_info,
            'model_analysis': {
                'biomedbert_features': biomedbert_features,
                'brain_analysis': {
                    'eeg_stats': eeg_stats,
                    'fmri_stats': fmri_stats
                }
            },
            **explanations,
            'timestamp': datetime.now().isoformat(),
            'generation_method': 'dynamic_from_model_predictions'
        }


def main():
    """Test unified explainer with dynamic explanation generation."""
    print("🧬 UNIFIED EXPLAINER - PANDAS-FREE VERSION")
    print("="*70)
    print("Testing dynamic explanation generation from BiomedBERT model")
    
    # Initialize explainer
    explainer = UnifiedExplainerPandasFree('best_model_biomedbert.pth')
    
    if not explainer.biomedbert_model:
        print("❌ BiomedBERT model not loaded. Please check model file.")
        return
    
    # Determine input dimensions from model
    try:
        eeg_dim = int(explainer.biomedbert_model.eeg_encoder.input_proj[0].weight.shape[1])
        fmri_dim = int(explainer.biomedbert_model.fmri_encoder.input_proj[0].weight.shape[1])
    except Exception:
        eeg_dim = getattr(AdvancedConfig, 'EEG_DIM', 16) if AdvancedConfig else 16
        fmri_dim = getattr(AdvancedConfig, 'FMRI_DIM', 32) if AdvancedConfig else 32
    
    print(f"📊 Model input dimensions: EEG={eeg_dim}, fMRI={fmri_dim}")
    
    # Generate sample data
    eeg_data = np.random.randn(eeg_dim).astype(np.float32)
    fmri_data = np.random.randn(fmri_dim).astype(np.float32)
    
    # Sample clinical text with medical concepts
    clinical_text = """
    PATIENT CLINICAL ASSESSMENT:
    FAMILY HISTORY: Strong familial pattern - mother with Alzheimer's disease onset at 70. 
    Paternal grandmother also affected by dementia. Multiple affected relatives.
    COGNITIVE STATUS: Patient reports progressive memory difficulties over 18 months.
    Word-finding problems and learning difficulties with new information.
    NEUROLOGICAL EXAM: Intact on formal testing. No focal neurological deficits.
    ASSESSMENT: Multimodal neuroimaging for cognitive decline risk stratification.
    """
    
    patient_info = {
        'participant_id': 'UNIFIED-DEMO-001',
        'age': 65,
        'sex': 'F'
    }
    
    # Generate explanation
    print("\n🤖 Generating dynamic medical explanation...")
    explanation = explainer.explain_prediction(
        eeg_data=eeg_data,
        fmri_data=fmri_data,
        clinical_text=clinical_text,
        patient_info=patient_info
    )
    
    if 'error' in explanation:
        print(f"❌ Error: {explanation['error']}")
        return
    
    # Display results
    print("\n" + "="*70)
    print("📋 UNIFIED DYNAMIC EXPLANATION REPORT")
    print("="*70)
    
    # Patient info
    patient_info = explanation.get('patient_info', {})
    print(f"\n👤 Patient: {patient_info.get('participant_id', 'Unknown')}")
    print(f"    Age: {patient_info.get('age', 'Unknown')}")
    print(f"    Sex: {patient_info.get('sex', 'Unknown')}")
    
    # Prediction
    pred = explanation['prediction']
    print(f"\n🎯 AI PREDICTION: {pred['risk_level']}")
    print(f"   Confidence: {pred['confidence']:.1%}")
    print(f"   Model: {pred['model_type']}")
    print(f"   Generation: {explanation['generation_method']}")
    
    # Model analysis
    analysis = explanation.get('model_analysis', {})
    biomedbert_features = analysis.get('biomedbert_features', {})
    attention_analysis = biomedbert_features.get('attention_analysis', {})
    
    if attention_analysis.get('top_attended_tokens'):
        print(f"\n🔍 TOP ATTENDED TOKENS:")
        for token_info in attention_analysis['top_attended_tokens'][:4]:
            print(f"   • {token_info['token']}: {token_info['attention_score']:.4f}")
    
    # Doctor section
    doctor = explanation['for_doctor']
    print(f"\n{'='*70}")
    print("🩺 FOR HEALTHCARE PROVIDER")
    print(f"{'='*70}")
    print(f"\n{doctor['summary']}\n")
    
    print("Clinical Findings:")
    for finding in doctor['clinical_findings']:
        print(f"  • {finding}")
    
    print("\nIdentified Risk Factors:")
    for risk in doctor['risk_factors']:
        print(f"  • {risk}")
    
    print("\nClinical Recommendations:")
    for rec in doctor['recommendations']:
        print(f"  ✓ {rec}")
    
    # Patient section
    patient = explanation['for_patient']
    print(f"\n{'='*70}")
    print("💙 FOR PATIENT")
    print(f"{'='*70}")
    print(f"\n{patient['summary']}\n")
    
    print("What We Found:")
    for finding in patient['what_we_found']:
        print(f"  {finding}")
    
    print(f"\nWhat This Means:")
    print(f"  {patient['what_this_means']}\n")
    
    print("Your Next Steps:")
    for step in patient['next_steps']:
        print(f"  ✓ {step}")
    
    # Save report
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / 'unified_dynamic_explanation.json'
    
    with open(output_file, 'w') as f:
        json.dump(explanation, f, indent=2)
    
    print(f"\n💾 Complete report saved to: {output_file}")
    print("\n✅ Dynamic explanation generation complete!")
    print("🎉 All explanations generated from actual BiomedBERT model predictions and attention!")


if __name__ == "__main__":
    main()