"""
Unified Hybrid Model + BiomedGPT Explainer
Generates explanations using actual BiomedGPT language model.
Combines deep learning predictions with AI-generated medical explanations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Import model architectures
try:
    from model_huggingface_fixed import AdvancedMultiModalNet, AdvancedConfig
    from transformers import AutoTokenizer, AutoModel
    HAS_BIOMEDBERT = True
except ImportError:
    HAS_BIOMEDBERT = False
    print("Warning: BiomedBERT model not available")

try:
    from hybrid_model_improved import ImprovedMultiModalNet, ImprovedConfig
    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False
    print("Warning: Hybrid model not available")

# Import BiomedGPT for explanation generation
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer as GPTTokenizer
    HAS_BIOMEDGPT = True
except ImportError:
    AutoModelForCausalLM = None
    GPTTokenizer = None
    HAS_BIOMEDGPT = False
    print("Warning: BiomedGPT not available")


class BiomedGPTExplainer:
    """Wrapper for BiomedGPT to generate medical explanations."""
    
    def __init__(
        self, 
        model_name: str = "stanford-crfm/BioMedLM",  # Alternative: "microsoft/BioGPT-Large"
        device: Optional[torch.device] = None
    ):
        """Initialize BiomedGPT model for explanation generation."""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        if HAS_BIOMEDGPT and GPTTokenizer is not None and AutoModelForCausalLM is not None:
            try:
                print(f"Loading BiomedGPT model: {model_name}")
                self.tokenizer = GPTTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
                
                self.model.eval()
                
                # Set pad token if not available
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                print(f"✅ BiomedGPT loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load BiomedGPT: {e}")
                print("Using fallback explanation generation")
    
    def generate_explanation(
        self, 
        prompt: str, 
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate medical explanation using BiomedGPT."""
        if not self.model or not self.tokenizer:
            return "BiomedGPT model not available. Using fallback explanation."
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            # Decode and extract generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return f"Error generating explanation: {str(e)}"


class UnifiedExplainer:
    """
    Unified explainer that combines predictions from hybrid deep learning model
    with BiomedGPT-generated medical explanations.
    """
    
    def __init__(
        self, 
        hybrid_model_path: str = 'best_model_fused.pth',
        biomedbert_model_path: str = 'best_model_biomedbert.pth',
        biomedgpt_model: str = "stanford-crfm/BioMedLM",
        device: Optional[torch.device] = None
    ):
        """Initialize unified explainer with both models and BiomedGPT."""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load prediction models
        self.hybrid_model = None
        if HAS_HYBRID and Path(hybrid_model_path).exists():
            try:
                self.hybrid_model = self._load_hybrid_model(hybrid_model_path)
                print(f"✅ Loaded hybrid model from {hybrid_model_path}")
            except Exception as e:
                print(f"❌ Failed to load hybrid model: {e}")
        
        self.biomedbert_model = None
        self.tokenizer = None
        self.hybrid_tokenizer = None
        if HAS_BIOMEDBERT and Path(biomedbert_model_path).exists():
            try:
                self.biomedbert_model, self.tokenizer = self._load_biomedbert_model(biomedbert_model_path)
                print(f"✅ Loaded BiomedBERT model from {biomedbert_model_path}")
            except Exception as e:
                print(f"❌ Failed to load BiomedBERT model: {e}")
        
        # Load tokenizer for hybrid model
        try:
            imp_cfg = globals().get('ImprovedConfig')
            hybrid_model_id = getattr(imp_cfg, 'TRANSFORMER_MODEL', None) if imp_cfg else None
            if hybrid_model_id:
                from transformers import AutoTokenizer as _ATok
                self.hybrid_tokenizer = _ATok.from_pretrained(hybrid_model_id)
        except Exception as e:
            print(f"Warning: Failed to load hybrid tokenizer: {e}")
        
        # Initialize BiomedGPT explainer
        self.gpt_explainer = BiomedGPTExplainer(model_name=biomedgpt_model, device=self.device)
    
    def _load_hybrid_model(self, path: str) -> Any:
        """Load hybrid deep learning model."""
        checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        
        eeg_dim = 500
        fmri_dim = 300
        try:
            if isinstance(state, dict):
                if 'eeg_encoder.input_proj.0.weight' in state:
                    eeg_dim = int(state['eeg_encoder.input_proj.0.weight'].shape[1])
                if 'fmri_encoder.input_proj.0.weight' in state:
                    fmri_dim = int(state['fmri_encoder.input_proj.0.weight'].shape[1])
        except Exception:
            pass
        
        imp_cfg = globals().get('ImprovedConfig')
        eeg_dim = getattr(imp_cfg, 'EEG_DIM', eeg_dim) if imp_cfg else eeg_dim
        fmri_dim = getattr(imp_cfg, 'FMRI_DIM', fmri_dim) if imp_cfg else fmri_dim
        
        model_cls = globals().get('ImprovedMultiModalNet')
        if model_cls is None:
            raise RuntimeError('ImprovedMultiModalNet class not available')
        model = model_cls(eeg_dim, fmri_dim).to(self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def _load_biomedbert_model(self, path: str) -> Tuple[Any, Optional[Any]]:
        """Load BiomedBERT-based model."""
        checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        
        eeg_dim = 500
        fmri_dim = 300
        try:
            if isinstance(state, dict):
                if 'eeg_encoder.input_proj.0.weight' in state:
                    eeg_dim = int(state['eeg_encoder.input_proj.0.weight'].shape[1])
                if 'fmri_encoder.input_proj.0.weight' in state:
                    fmri_dim = int(state['fmri_encoder.input_proj.0.weight'].shape[1])
        except Exception:
            pass
        
        adv_cfg = globals().get('AdvancedConfig')
        eeg_dim = getattr(adv_cfg, 'EEG_DIM', eeg_dim) if adv_cfg else eeg_dim
        fmri_dim = getattr(adv_cfg, 'FMRI_DIM', fmri_dim) if adv_cfg else fmri_dim
        model_name = getattr(adv_cfg, 'CHOSEN_MODEL', 'biomedbert') if adv_cfg else 'biomedbert'
        
        model_cls = globals().get('AdvancedMultiModalNet')
        if model_cls is None:
            raise RuntimeError('AdvancedMultiModalNet class not available')
        model = model_cls(eeg_dim=eeg_dim, fmri_dim=fmri_dim, model_name=model_name).to(self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        try:
            if 'AutoTokenizer' in globals():
                adv_cfg = globals().get('AdvancedConfig')
                hf_id = None
                if adv_cfg:
                    hf_id = getattr(adv_cfg, 'AVAILABLE_MODELS', {}).get(model_name, None)
                from transformers import AutoTokenizer as _ATok
                tokenizer = _ATok.from_pretrained(hf_id or model_name)
            else:
                tokenizer = None
        except Exception:
            tokenizer = None
        
        return model, tokenizer

    def get_model_input_dims(self) -> Tuple[int, int]:
        """Infer input dimensions from loaded models or defaults."""
        for m in [self.biomedbert_model, self.hybrid_model]:
            if m is not None:
                try:
                    w_eeg = m.eeg_encoder.input_proj[0].weight
                    w_fmri = m.fmri_encoder.input_proj[0].weight
                    return int(w_eeg.shape[1]), int(w_fmri.shape[1])
                except Exception:
                    continue
        
        adv_cfg = globals().get('AdvancedConfig')
        imp_cfg = globals().get('ImprovedConfig')
        eeg_dim = getattr(adv_cfg, 'EEG_DIM', None) or getattr(imp_cfg, 'EEG_DIM', None) or 500
        fmri_dim = getattr(adv_cfg, 'FMRI_DIM', None) or getattr(imp_cfg, 'FMRI_DIM', None) or 300
        return int(eeg_dim), int(fmri_dim)
    
    def _construct_biomedgpt_prompt(
        self,
        prediction: int,
        confidence: float,
        probabilities: List[float],
        eeg_features: Dict[str, Any],
        fmri_features: Dict[str, Any],
        clinical_text: str,
        model_agreement: bool,
        audience: str = "doctor"
    ) -> str:
        """Construct prompt for BiomedGPT based on model outputs."""
        
        risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
        
        if audience == "doctor":
            prompt = f"""As a medical AI assistant, provide a clinical explanation for the following dementia risk assessment:

ASSESSMENT RESULTS:
- Risk Classification: {risk_level}
- Model Confidence: {confidence:.1%}
- Probability Distribution: Low-risk {probabilities[0]:.3f}, High-risk {probabilities[1]:.3f}
- Model Agreement: {"Models agree" if model_agreement else "Models disagree"}

MULTIMODAL FINDINGS:
- EEG Analysis: Mean activity {eeg_features['mean_activity']:.3f}, Variability {eeg_features['variability']:.3f}, Pattern: {eeg_features['interpretation']}
- fMRI Analysis: Mean connectivity {fmri_features['mean_connectivity']:.3f}, Status: {fmri_features['connectivity_status']}

CLINICAL CONTEXT:
{clinical_text}

Provide a concise clinical summary including:
1. Interpretation of multimodal findings
2. Key risk factors identified
3. Clinical recommendations for follow-up

Response:"""
        
        else:  # patient
            prompt = f"""As a medical AI assistant, explain the following dementia risk assessment results to a patient in simple, clear language:

ASSESSMENT RESULTS:
- Risk Level: {risk_level}
- Confidence: {confidence:.0%}

KEY FINDINGS:
- Brain Activity (EEG): {eeg_features['interpretation']}
- Brain Networks (fMRI): {fmri_features['connectivity_status']} connectivity

Provide a patient-friendly explanation including:
1. What these results mean in simple terms
2. What was found in the brain scans
3. Practical next steps and lifestyle recommendations

Use simple language, avoid medical jargon, and be reassuring while honest. Response:"""
        
        return prompt
    
    def generate_unified_explanation(
        self,
        eeg_data: np.ndarray,
        fmri_data: np.ndarray,
        clinical_text: str,
        patient_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate unified explanation using BiomedGPT.
        Returns explanations for both doctors and patients.
        """
        
        patient_info = patient_info or {}
        participant_id = patient_info.get('participant_id', 'Unknown')
        age = patient_info.get('age', 'N/A')
        sex = patient_info.get('sex', 'N/A')
        
        print(f"\n🔬 Generating Unified Explanation for Patient {participant_id}")
        print("=" * 70)
        
        # Convert inputs to tensors
        eeg_tensor = torch.FloatTensor(eeg_data).unsqueeze(0).to(self.device)
        fmri_tensor = torch.FloatTensor(fmri_data).unsqueeze(0).to(self.device)
        
        # Tokenize clinical text
        text_inputs_biomed = None
        text_inputs_hybrid = None
        if self.tokenizer:
            adv_cfg = globals().get('AdvancedConfig')
            max_len = getattr(adv_cfg, 'MAX_TEXT_LENGTH', 256) if adv_cfg else 256
            text_inputs_biomed = self.tokenizer(
                clinical_text,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            text_inputs_biomed = {k: v.to(self.device) for k, v in text_inputs_biomed.items()}
        
        if self.hybrid_tokenizer:
            imp_cfg = globals().get('ImprovedConfig')
            max_len_h = getattr(imp_cfg, 'MAX_TEXT_LENGTH', 256) if imp_cfg else 256
            text_inputs_hybrid = self.hybrid_tokenizer(
                clinical_text,
                max_length=max_len_h,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            text_inputs_hybrid = {k: v.to(self.device) for k, v in text_inputs_hybrid.items()}
        
        # Get predictions from both models
        hybrid_prediction = self._get_hybrid_prediction(eeg_tensor, fmri_tensor, text_inputs_hybrid)
        biomedbert_prediction = self._get_biomedbert_prediction(eeg_tensor, fmri_tensor, text_inputs_biomed)
        
        # Ensemble predictions
        ensemble_result = self._ensemble_predictions(hybrid_prediction, biomedbert_prediction)
        
        # Extract features for explanation
        eeg_features = self._analyze_eeg_patterns(eeg_data)
        fmri_features = self._analyze_fmri_connectivity(fmri_data)
        
        print("\n🤖 Generating AI explanations with BiomedGPT...")
        
        # Generate doctor explanation with BiomedGPT
        doctor_prompt = self._construct_biomedgpt_prompt(
            prediction=ensemble_result['prediction'],
            confidence=ensemble_result['confidence'],
            probabilities=ensemble_result.get('probabilities', [0.5, 0.5]),
            eeg_features=eeg_features,
            fmri_features=fmri_features,
            clinical_text=clinical_text,
            model_agreement=ensemble_result.get('models_agree', True),
            audience="doctor"
        )
        
        doctor_explanation = self.gpt_explainer.generate_explanation(
            prompt=doctor_prompt,
            max_length=600,
            temperature=0.7
        )
        
        # Generate patient explanation with BiomedGPT
        patient_prompt = self._construct_biomedgpt_prompt(
            prediction=ensemble_result['prediction'],
            confidence=ensemble_result['confidence'],
            probabilities=ensemble_result.get('probabilities', [0.5, 0.5]),
            eeg_features=eeg_features,
            fmri_features=fmri_features,
            clinical_text=clinical_text,
            model_agreement=ensemble_result.get('models_agree', True),
            audience="patient"
        )
        
        patient_explanation = self.gpt_explainer.generate_explanation(
            prompt=patient_prompt,
            max_length=500,
            temperature=0.7
        )
        
        # Return structured explanation
        return {
            'patient_info': patient_info,
            'prediction': {
                'risk_level': 'HIGH RISK' if ensemble_result['prediction'] == 1 else 'LOW RISK',
                'confidence': ensemble_result['confidence'],
                'prediction_class': ensemble_result['prediction'],
                'ensemble_method': ensemble_result.get('ensemble_method'),
                'models_agree': ensemble_result.get('models_agree', None),
                'probabilities': ensemble_result.get('probabilities', [])
            },
            'for_doctor': {
                'ai_generated_explanation': doctor_explanation,
                'technical_details': {
                    'model_confidence': {
                        'hybrid_dl': ensemble_result.get('hybrid_confidence', 'N/A'),
                        'biomedbert': ensemble_result.get('biomedbert_confidence', 'N/A'),
                        'ensemble': ensemble_result['confidence']
                    },
                    'probability_distribution': ensemble_result.get('probabilities', []),
                    'feature_analysis': {
                        'eeg_variability': eeg_features['variability'],
                        'fmri_connectivity': fmri_features['mean_connectivity']
                    }
                }
            },
            'for_patient': {
                'ai_generated_explanation': patient_explanation
            },
            'modality_analysis': {
                'eeg': eeg_features,
                'fmri': fmri_features,
                'clinical_text': clinical_text
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_hybrid_prediction(
        self, 
        eeg: torch.Tensor, 
        fmri: torch.Tensor,
        text_inputs: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Get prediction from hybrid model."""
        if not self.hybrid_model or text_inputs is None:
            return {'available': False}
        
        with torch.no_grad():
            try:
                output = self.hybrid_model(eeg, fmri, text_inputs)
                probs = F.softmax(output, dim=1)
                pred_class = int(torch.argmax(probs, dim=1).item())
                confidence = float(probs[0][pred_class].item())
                
                return {
                    'available': True,
                    'prediction': pred_class,
                    'confidence': confidence,
                    'probabilities': probs[0].detach().cpu().numpy().tolist(),
                    'model_type': 'hybrid_deep_learning'
                }
            except Exception as e:
                print(f"Hybrid model prediction failed: {e}")
                return {'available': False}
    
    def _get_biomedbert_prediction(
        self,
        eeg: torch.Tensor,
        fmri: torch.Tensor,
        text_inputs: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Get prediction from BiomedBERT model."""
        if not self.biomedbert_model or text_inputs is None:
            return {'available': False}
        
        with torch.no_grad():
            try:
                output = self.biomedbert_model(eeg, fmri, text_inputs)
                probs = F.softmax(output, dim=1)
                pred_class = int(torch.argmax(probs, dim=1).item())
                confidence = float(probs[0][pred_class].item())
                
                return {
                    'available': True,
                    'prediction': pred_class,
                    'confidence': confidence,
                    'probabilities': probs[0].detach().cpu().numpy().tolist(),
                    'model_type': 'biomedbert_medical'
                }
            except Exception as e:
                print(f"BiomedBERT model prediction failed: {e}")
                return {'available': False}
    
    def _ensemble_predictions(
        self,
        hybrid_pred: Dict[str, Any],
        biomedbert_pred: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ensemble predictions from both models."""
        
        if hybrid_pred['available'] and not biomedbert_pred['available']:
            return {**hybrid_pred, 'ensemble_method': 'hybrid_only'}
        elif biomedbert_pred['available'] and not hybrid_pred['available']:
            return {**biomedbert_pred, 'ensemble_method': 'biomedbert_only'}
        elif not hybrid_pred['available'] and not biomedbert_pred['available']:
            return {'available': False, 'ensemble_method': 'none'}
        
        # Both models available - ensemble
        hybrid_probs = np.array(hybrid_pred['probabilities'])
        biomedbert_probs = np.array(biomedbert_pred['probabilities'])
        
        # Weighted average (give more weight to BiomedBERT for medical domain)
        ensemble_probs = 0.4 * hybrid_probs + 0.6 * biomedbert_probs
        ensemble_class = int(np.argmax(ensemble_probs))
        ensemble_confidence = float(ensemble_probs[ensemble_class])
        
        # Calculate agreement
        agreement = hybrid_pred['prediction'] == biomedbert_pred['prediction']
        
        return {
            'available': True,
            'prediction': ensemble_class,
            'confidence': ensemble_confidence,
            'probabilities': ensemble_probs.tolist(),
            'ensemble_method': 'weighted_average',
            'models_agree': agreement,
            'hybrid_confidence': hybrid_pred['confidence'],
            'biomedbert_confidence': biomedbert_pred['confidence']
        }
    
    def _analyze_eeg_patterns(self, eeg_data: np.ndarray) -> Dict[str, Any]:
        """Analyze EEG patterns for explanation."""
        mean_activity = float(np.mean(eeg_data))
        std_activity = float(np.std(eeg_data))
        
        abnormal = std_activity > np.abs(mean_activity) * 2
        
        return {
            'mean_activity': mean_activity,
            'variability': std_activity,
            'abnormal_patterns': abnormal,
            'interpretation': 'irregular' if abnormal else 'within normal range'
        }
    
    def _analyze_fmri_connectivity(self, fmri_data: np.ndarray) -> Dict[str, Any]:
        """Analyze fMRI connectivity for explanation."""
        mean_connectivity = float(np.mean(fmri_data))
        reduced_connectivity = mean_connectivity < -0.1
        
        return {
            'mean_connectivity': mean_connectivity,
            'connectivity_status': 'reduced' if reduced_connectivity else 'normal',
            'clinical_relevance': 'high' if reduced_connectivity else 'medium'
        }
    
    def save_explanation_report(self, explanation: Dict[str, Any], output_path: str):
        """Save explanation to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_file, 'w') as f:
            json.dump(explanation, f, indent=2)
        
        print(f"\n💾 Explanation report saved to: {output_path}")
    
    def print_explanation(self, explanation: Dict[str, Any]):
        """Print formatted explanation to console."""
        
        if 'error' in explanation:
            print(f"\n❌ {explanation['error']}")
            return
        
        print("\n" + "="*70)
        print("📋 AI-GENERATED MEDICAL EXPLANATION REPORT")
        print("="*70)
        
        # Patient info
        patient_info = explanation.get('patient_info', {})
        print(f"\n👤 Patient: {patient_info.get('participant_id', 'N/A')}")
        print(f"   Age: {patient_info.get('age', 'N/A')} | Sex: {patient_info.get('sex', 'N/A')}")
        
        # Prediction
        pred = explanation['prediction']
        print(f"\n🎯 PREDICTION: {pred['risk_level']}")
        print(f"   Confidence: {pred['confidence']:.1%}")
        print(f"   Ensemble Method: {pred['ensemble_method']}")
        if pred.get('models_agree') is not None:
            print(f"   Models Agree: {'✅ Yes' if pred['models_agree'] else '⚠️ No'}")
        
        # Doctor section
        doctor = explanation['for_doctor']
        print(f"\n{'='*70}")
        print("🩺 FOR HEALTHCARE PROVIDER (BiomedGPT Generated)")
        print(f"{'='*70}")
        print(f"\n{doctor['ai_generated_explanation']}\n")
        
        # Patient section
        patient = explanation['for_patient']
        print(f"\n{'='*70}")
        print("💙 FOR PATIENT (BiomedGPT Generated)")
        print(f"{'='*70}")
        print(f"\n{patient['ai_generated_explanation']}\n")
        
        print("="*70)


def main():
    """Demo of unified explainer with BiomedGPT."""
    print("🧬 UNIFIED EXPLAINER WITH BIOMEDGPT")
    print("="*70)
    print("Using Real AI Language Model for Medical Explanations\n")
    
    # Initialize explainer with BiomedGPT
    explainer = UnifiedExplainer(
        hybrid_model_path='best_model_fused.pth',
        biomedbert_model_path='best_model_biomedbert.pth',
        biomedgpt_model="stanford-crfm/BioMedLM"  # or "microsoft/BioGPT-Large"
    )
    
    # Create sample data
    print("\n📊 Generating sample patient data...")
    eeg_dim, fmri_dim = explainer.get_model_input_dims()
    print(f"Using dimensions: EEG={eeg_dim}, fMRI={fmri_dim}")
    
    eeg_data = np.random.randn(eeg_dim).astype(np.float32)
    fmri_data = np.random.randn(fmri_dim).astype(np.float32)
    
    clinical_text = """
    PATIENT CLINICAL HISTORY:
    FAMILY HISTORY: Mother diagnosed with Alzheimer's disease at age 72. 
    Father has mild cognitive impairment.
    COGNITIVE PROFILE: Patient reports occasional memory difficulties and 
    word-finding issues. Documented learning difficulties in early education.
    CURRENT STATUS: Neurologically intact on examination. No focal deficits.
    ASSESSMENT: Multimodal neuroimaging evaluation for cognitive decline risk.
    """
    
    patient_info = {
        'participant_id': 'SUB-001',
        'age': 68,
        'sex': 'F'
    }
    
    # Generate AI explanation
    explanation = explainer.generate_unified_explanation(
        eeg_data=eeg_data,
        fmri_data=fmri_data,
        clinical_text=clinical_text,
        patient_info=patient_info
    )
    
    # Display explanation
    explainer.print_explanation(explanation)
    
    # Save report
    explainer.save_explanation_report(
        explanation,
        'results/biomedgpt_explanation_report.json'
    )
    
    print("\n✅ AI explanation generation complete!")


if __name__ == "__main__":
    main()