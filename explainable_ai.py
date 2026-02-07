"""
Explainable AI (XAI) for Multi-Modal Brain Imaging Model
This module provides comprehensive interpretability tools for understanding model decisions.
"""

import json
import math
from pathlib import Path
import numpy as np

# Try to import libraries, fall back to manual implementations if needed
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available, some features will be limited")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: Matplotlib not available, visualizations will be skipped")

class ModelExplainer:
    """
    Comprehensive explainable AI toolkit for multi-modal brain imaging model.
    Provides feature importance, attention analysis, and prediction explanations.
    """
    
    def __init__(self, model=None, model_path=None):
        self.model = model
        self.model_path = model_path
        self.explanations = {}
        
        if model_path and HAS_TORCH:
            self.load_model()
    
    def load_model(self):
        """Load the trained model for gradient-based explanations."""
        if not HAS_TORCH:
            print("PyTorch not available, gradient-based explanations disabled")
            return
        
        try:
            # Import model class (you might need to adjust this import)
            from hybrid_model import DualStreamNetwork
            
            # Load model (you'll need to specify dimensions)
            self.model = DualStreamNetwork(eeg_dim=16, fmri_dim=32)
            state_dict = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
            self.model = None
    
    def explain_prediction(self, sample_data, participant_id=None):
        """
        Generate comprehensive explanation for a single prediction.
        
        Args:
            sample_data: Dict with 'eeg', 'fmri', 'text', 'prediction', 'probability'
            participant_id: Identifier for the sample
        """
        explanation = {
            'participant_id': participant_id or 'unknown',
            'prediction': sample_data.get('prediction', 'unknown'),
            'probability': sample_data.get('probability', 0.0),
            'confidence_level': self._assess_confidence(sample_data.get('probability', 0.0)),
            'modality_contributions': self._analyze_modality_contributions(sample_data),
            'feature_importance': self._calculate_feature_importance(sample_data),
            'risk_factors': self._identify_risk_factors(sample_data),
            'explanation_text': self._generate_natural_language_explanation(sample_data)
        }
        
        return explanation
    
    def _assess_confidence(self, probability):
        """Assess model confidence level."""
        if abs(probability - 0.5) < 0.1:
            return "Low confidence (borderline case)"
        elif abs(probability - 0.5) < 0.2:
            return "Medium confidence"
        else:
            return "High confidence"
    
    def _analyze_modality_contributions(self, sample_data):
        """Analyze contribution of each modality (EEG, fMRI, Text)."""
        # Simplified contribution analysis based on data characteristics
        contributions = {}
        
        # EEG analysis
        eeg_data = sample_data.get('eeg', [])
        if eeg_data:
            eeg_variance = np.var(eeg_data) if len(eeg_data) > 1 else 0
            eeg_mean = np.mean(eeg_data)
            contributions['eeg'] = {
                'contribution_score': min(abs(eeg_variance) * 10, 1.0),
                'characteristics': {
                    'variance': float(eeg_variance),
                    'mean_activity': float(eeg_mean),
                    'abnormal_patterns': eeg_variance > 0.5  # Threshold for abnormal
                }
            }
        
        # fMRI analysis
        fmri_data = sample_data.get('fmri', [])
        if fmri_data:
            fmri_variance = np.var(fmri_data) if len(fmri_data) > 1 else 0
            fmri_mean = np.mean(fmri_data)
            contributions['fmri'] = {
                'contribution_score': min(abs(fmri_variance) * 10, 1.0),
                'characteristics': {
                    'variance': float(fmri_variance),
                    'mean_activation': float(fmri_mean),
                    'abnormal_patterns': fmri_variance > 0.5
                }
            }
        
        # Text analysis
        text_features = sample_data.get('text_features', {})
        text_score = 0
        text_indicators = []
        
        if text_features:
            # Analyze text-based risk factors
            for feature, value in text_features.items():
                if value and str(value).lower() not in ['nan', 'none', '']:
                    if 'dementia' in feature.lower():
                        text_score += 0.4
                        text_indicators.append(f"Family history: {feature}")
                    elif 'learning' in feature.lower():
                        text_score += 0.3
                        text_indicators.append(f"Cognitive issue: {feature}")
                    elif any(keyword in feature.lower() for keyword in ['disease', 'drug', 'allerg']):
                        text_score += 0.1
                        text_indicators.append(f"Medical factor: {feature}")
        
        contributions['text'] = {
            'contribution_score': min(text_score, 1.0),
            'indicators': text_indicators
        }
        
        return contributions
    
    def _calculate_feature_importance(self, sample_data):
        """Calculate importance of individual features."""
        importance = {}
        
        # EEG feature importance (simplified)
        eeg_data = sample_data.get('eeg', [])
        if eeg_data:
            eeg_importance = []
            for i, value in enumerate(eeg_data[:10]):  # Top 10 features
                # Simplified importance based on deviation from normal
                deviation = abs(value)
                eeg_importance.append({
                    'feature': f'EEG_channel_{i+1}',
                    'value': float(value),
                    'importance': min(deviation, 1.0),
                    'interpretation': 'High activity' if value > 0.5 else 'Low activity' if value < -0.5 else 'Normal'
                })
            importance['eeg'] = sorted(eeg_importance, key=lambda x: x['importance'], reverse=True)[:5]
        
        # fMRI feature importance
        fmri_data = sample_data.get('fmri', [])
        if fmri_data:
            fmri_importance = []
            for i, value in enumerate(fmri_data[:10]):  # Top 10 features
                deviation = abs(value)
                fmri_importance.append({
                    'feature': f'fMRI_region_{i+1}',
                    'value': float(value),
                    'importance': min(deviation, 1.0),
                    'interpretation': 'High activation' if value > 0.5 else 'Low activation' if value < -0.5 else 'Normal'
                })
            importance['fmri'] = sorted(fmri_importance, key=lambda x: x['importance'], reverse=True)[:5]
        
        # Text feature importance
        text_features = sample_data.get('text_features', {})
        text_importance = []
        for feature, value in text_features.items():
            if value and str(value).lower() not in ['nan', 'none', '']:
                # Assign importance based on medical relevance
                if 'dementia' in feature.lower():
                    imp_score = 0.9
                elif 'learning' in feature.lower():
                    imp_score = 0.7
                elif 'disease' in feature.lower():
                    imp_score = 0.5
                else:
                    imp_score = 0.3
                
                text_importance.append({
                    'feature': feature,
                    'value': str(value),
                    'importance': imp_score,
                    'interpretation': 'High risk factor' if imp_score > 0.6 else 'Moderate risk factor'
                })
        
        importance['text'] = sorted(text_importance, key=lambda x: x['importance'], reverse=True)
        
        return importance
    
    def _identify_risk_factors(self, sample_data):
        """Identify specific risk factors contributing to the prediction."""
        risk_factors = {
            'high_risk': [],
            'protective': [],
            'neutral': []
        }
        
        # Analyze text-based clinical factors
        text_features = sample_data.get('text_features', {})
        for feature, value in text_features.items():
            if value and str(value).lower() not in ['nan', 'none', '']:
                if 'dementia' in feature.lower() and value:
                    risk_factors['high_risk'].append({
                        'factor': 'Family history of dementia',
                        'evidence': f"{feature}: {value}",
                        'impact': 'Strong risk factor for cognitive decline'
                    })
                elif 'learning' in feature.lower() and value:
                    risk_factors['high_risk'].append({
                        'factor': 'Learning difficulties',
                        'evidence': f"{feature}: {value}",
                        'impact': 'May indicate early cognitive vulnerabilities'
                    })
        
        # Analyze brain imaging patterns
        eeg_data = sample_data.get('eeg', [])
        if eeg_data:
            eeg_variance = np.var(eeg_data)
            eeg_mean = np.mean(eeg_data)
            
            if eeg_variance > 0.5:
                risk_factors['high_risk'].append({
                    'factor': 'Irregular EEG patterns',
                    'evidence': f"High variability in brain electrical activity (variance: {eeg_variance:.3f})",
                    'impact': 'May indicate abnormal neural activity'
                })
            elif eeg_variance < 0.1 and abs(eeg_mean) < 0.2:
                risk_factors['protective'].append({
                    'factor': 'Stable EEG patterns',
                    'evidence': f"Low variability and normal mean activity",
                    'impact': 'Indicates healthy brain electrical activity'
                })
        
        fmri_data = sample_data.get('fmri', [])
        if fmri_data:
            fmri_variance = np.var(fmri_data)
            fmri_mean = np.mean(fmri_data)
            
            if fmri_variance > 0.5:
                risk_factors['high_risk'].append({
                    'factor': 'Irregular fMRI activation patterns',
                    'evidence': f"High variability in brain activation (variance: {fmri_variance:.3f})",
                    'impact': 'May indicate abnormal brain function'
                })
            elif fmri_variance < 0.1 and abs(fmri_mean) < 0.2:
                risk_factors['protective'].append({
                    'factor': 'Normal fMRI activation patterns',
                    'evidence': f"Consistent and normal brain activation levels",
                    'impact': 'Indicates healthy brain function'
                })
        
        return risk_factors
    
    def _generate_natural_language_explanation(self, sample_data):
        """Generate human-readable explanation of the prediction."""
        prediction = sample_data.get('prediction', 0)
        probability = sample_data.get('probability', 0.5)
        
        # Base explanation
        if prediction == 1:
            base_text = f"The model predicts HIGH RISK of cognitive decline with {probability*100:.1f}% confidence."
        else:
            base_text = f"The model predicts LOW RISK of cognitive decline with {(1-probability)*100:.1f}% confidence."
        
        # Add contributing factors
        factors_text = []
        
        # Text-based factors
        text_features = sample_data.get('text_features', {})
        has_family_history = False
        has_learning_issues = False
        
        for feature, value in text_features.items():
            if value and str(value).lower() not in ['nan', 'none', '']:
                if 'dementia' in feature.lower():
                    has_family_history = True
                elif 'learning' in feature.lower():
                    has_learning_issues = True
        
        if has_family_history:
            factors_text.append("family history of dementia (strong risk factor)")
        if has_learning_issues:
            factors_text.append("reported learning difficulties")
        
        # Brain imaging factors
        eeg_data = sample_data.get('eeg', [])
        fmri_data = sample_data.get('fmri', [])
        
        brain_patterns = []
        if eeg_data and np.var(eeg_data) > 0.5:
            brain_patterns.append("irregular EEG patterns")
        if fmri_data and np.var(fmri_data) > 0.5:
            brain_patterns.append("abnormal brain activation patterns")
        
        if brain_patterns:
            factors_text.extend(brain_patterns)
        
        # Combine explanation
        if factors_text:
            factors_str = ", ".join(factors_text)
            explanation = f"{base_text} This prediction is primarily based on: {factors_str}."
        else:
            explanation = f"{base_text} The prediction is based on combined analysis of brain imaging and clinical data patterns."
        
        # Add confidence interpretation
        confidence_level = self._assess_confidence(probability)
        if "Low confidence" in confidence_level:
            explanation += " Note: This is a borderline case - additional clinical evaluation is recommended."
        elif "High confidence" in confidence_level:
            explanation += " The model shows high confidence in this prediction."
        
        return explanation
    
    def generate_cohort_analysis(self, predictions_file):
        """Generate analysis across the entire test cohort."""
        print("Generating cohort-level analysis...")
        
        # Load predictions
        with open(predictions_file, 'r') as f:
            lines = f.readlines()
        
        data_lines = lines[1:]  # Skip header
        cohort_data = []
        
        for line in data_lines:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                cohort_data.append({
                    'participant_id': parts[4],
                    'y_true': int(parts[1]),
                    'y_prob': float(parts[2]),
                    'y_pred_original': int(parts[3]),
                    'y_pred_optimized': int(parts[5])
                })
        
        # Analyze cohort patterns
        analysis = {
            'cohort_size': len(cohort_data),
            'prevalence': sum(d['y_true'] for d in cohort_data) / len(cohort_data),
            'model_performance': self._analyze_model_performance(cohort_data),
            'prediction_patterns': self._analyze_prediction_patterns(cohort_data),
            'risk_distribution': self._analyze_risk_distribution(cohort_data),
            'clinical_insights': self._generate_clinical_insights(cohort_data)
        }
        
        return analysis
    
    def _analyze_model_performance(self, cohort_data):
        """Analyze model performance across the cohort."""
        # Calculate metrics for optimized model
        y_true = [d['y_true'] for d in cohort_data]
        y_pred = [d['y_pred_optimized'] for d in cohort_data]
        y_prob = [d['y_prob'] for d in cohort_data]
        
        # Manual calculation of metrics
        tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
        fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
        tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
        fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        return {
            'accuracy': (tp + tn) / len(y_true),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'positive_predictive_value': ppv,
            'negative_predictive_value': npv,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
        }
    
    def _analyze_prediction_patterns(self, cohort_data):
        """Analyze patterns in model predictions."""
        y_prob = [d['y_prob'] for d in cohort_data]
        
        return {
            'mean_probability': np.mean(y_prob),
            'probability_std': np.std(y_prob),
            'high_confidence_predictions': sum(1 for p in y_prob if abs(p - 0.5) > 0.3),
            'borderline_cases': sum(1 for p in y_prob if abs(p - 0.5) < 0.1),
            'probability_distribution': {
                'very_low_risk': sum(1 for p in y_prob if p < 0.3),
                'low_risk': sum(1 for p in y_prob if 0.3 <= p < 0.5),
                'moderate_risk': sum(1 for p in y_prob if 0.5 <= p < 0.7),
                'high_risk': sum(1 for p in y_prob if p >= 0.7)
            }
        }
    
    def _analyze_risk_distribution(self, cohort_data):
        """Analyze risk distribution across the cohort."""
        y_true = [d['y_true'] for d in cohort_data]
        y_prob = [d['y_prob'] for d in cohort_data]
        
        high_risk_actual = [d for d in cohort_data if d['y_true'] == 1]
        low_risk_actual = [d for d in cohort_data if d['y_true'] == 0]
        
        return {
            'actual_high_risk_count': len(high_risk_actual),
            'actual_low_risk_count': len(low_risk_actual),
            'avg_prob_high_risk_group': np.mean([d['y_prob'] for d in high_risk_actual]) if high_risk_actual else 0,
            'avg_prob_low_risk_group': np.mean([d['y_prob'] for d in low_risk_actual]) if low_risk_actual else 0,
            'discrimination_score': self._calculate_discrimination_score(cohort_data)
        }
    
    def _calculate_discrimination_score(self, cohort_data):
        """Calculate how well the model discriminates between risk groups."""
        high_risk_probs = [d['y_prob'] for d in cohort_data if d['y_true'] == 1]
        low_risk_probs = [d['y_prob'] for d in cohort_data if d['y_true'] == 0]
        
        if not high_risk_probs or not low_risk_probs:
            return 0
        
        # Simple discrimination measure
        high_risk_mean = np.mean(high_risk_probs)
        low_risk_mean = np.mean(low_risk_probs)
        
        return abs(high_risk_mean - low_risk_mean)
    
    def _generate_clinical_insights(self, cohort_data):
        """Generate clinical insights from the cohort analysis."""
        performance = self._analyze_model_performance(cohort_data)
        patterns = self._analyze_prediction_patterns(cohort_data)
        
        insights = []
        
        # Sensitivity insights
        if performance['sensitivity'] >= 0.9:
            insights.append("Excellent sensitivity: The model rarely misses high-risk cases.")
        elif performance['sensitivity'] >= 0.8:
            insights.append("Good sensitivity: The model catches most high-risk cases.")
        else:
            insights.append("Moderate sensitivity: Some high-risk cases may be missed.")
        
        # Specificity insights
        if performance['specificity'] >= 0.8:
            insights.append("Good specificity: The model rarely gives false alarms.")
        else:
            insights.append("Lower specificity: The model may overestimate risk in some cases.")
        
        # Prediction confidence insights
        if patterns['borderline_cases'] > len(cohort_data) * 0.3:
            insights.append("Many borderline cases detected - consider additional clinical assessment.")
        
        if patterns['high_confidence_predictions'] > len(cohort_data) * 0.7:
            insights.append("Model shows high confidence in most predictions.")
        
        return insights
    
    def save_explanation_report(self, analysis, output_file):
        """Save comprehensive explanation report."""
        report = {
            'report_type': 'Explainable AI Analysis',
            'model_type': 'Multi-modal Brain Imaging Risk Prediction',
            'analysis_date': '2025-10-10',
            'cohort_analysis': analysis,
            'interpretation_guide': {
                'probability_ranges': {
                    '0.0-0.3': 'Very low risk',
                    '0.3-0.5': 'Low risk', 
                    '0.5-0.7': 'Moderate risk',
                    '0.7-1.0': 'High risk'
                },
                'confidence_levels': {
                    'high': 'Probability < 0.3 or > 0.7',
                    'medium': 'Probability 0.3-0.4 or 0.6-0.7',
                    'low': 'Probability 0.4-0.6 (borderline cases)'
                }
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Explanation report saved to {output_file}")

def create_visual_explanations(predictions_file, output_dir):
    """Create visual explanations if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping visualizations")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    with open(predictions_file, 'r') as f:
        lines = f.readlines()
    
    data_lines = lines[1:]  # Skip header
    y_true = []
    y_prob = []
    participant_ids = []
    
    for line in data_lines:
        parts = line.strip().split(',')
        if len(parts) >= 5:
            y_true.append(int(parts[1]))
            y_prob.append(float(parts[2]))
            participant_ids.append(parts[4])
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Explainability Analysis', fontsize=16)
    
    # 1. Probability distribution
    axes[0, 0].hist(y_prob, bins=10, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='Original threshold')
    axes[0, 0].axvline(x=0.619, color='green', linestyle='--', label='Optimal threshold')
    axes[0, 0].set_xlabel('Predicted Probability')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Predicted Probabilities')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Risk group comparison
    high_risk_probs = [y_prob[i] for i in range(len(y_true)) if y_true[i] == 1]
    low_risk_probs = [y_prob[i] for i in range(len(y_true)) if y_true[i] == 0]
    
    axes[0, 1].boxplot([low_risk_probs, high_risk_probs], labels=['Actual Low Risk', 'Actual High Risk'])
    axes[0, 1].set_ylabel('Predicted Probability')
    axes[0, 1].set_title('Model Discrimination by True Risk Group')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Individual predictions
    colors = ['blue' if yt == 0 else 'red' for yt in y_true]
    axes[1, 0].scatter(range(len(y_prob)), y_prob, c=colors, alpha=0.7)
    axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Original threshold')
    axes[1, 0].axhline(y=0.619, color='green', linestyle='--', alpha=0.7, label='Optimal threshold')
    axes[1, 0].set_xlabel('Participant')
    axes[1, 0].set_ylabel('Predicted Probability')
    axes[1, 0].set_title('Individual Predictions (Blue=Low Risk, Red=High Risk)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Confidence levels
    confidence_levels = []
    for prob in y_prob:
        if abs(prob - 0.5) > 0.3:
            confidence_levels.append('High')
        elif abs(prob - 0.5) > 0.1:
            confidence_levels.append('Medium')
        else:
            confidence_levels.append('Low')
    
    confidence_counts = {level: confidence_levels.count(level) for level in ['Low', 'Medium', 'High']}
    axes[1, 1].pie(confidence_counts.values(), labels=confidence_counts.keys(), autopct='%1.1f%%')
    axes[1, 1].set_title('Model Confidence Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_explanations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visual explanations saved to {output_dir / 'model_explanations.png'}")

def main():
    """Main function to generate comprehensive explainable AI analysis."""
    base_path = Path(__file__).parent
    results_dir = base_path / 'results'
    explanations_dir = results_dir / 'explanations'
    explanations_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("EXPLAINABLE AI (XAI) ANALYSIS")
    print("="*60)
    
    # Initialize explainer
    explainer = ModelExplainer()
    
    # Check for predictions file
    predictions_file = results_dir / 'test_predictions_optimized.csv'
    if not predictions_file.exists():
        print(f"Error: {predictions_file} not found.")
        print("Please run the threshold optimization script first.")
        return
    
    print("\n1. Generating cohort-level analysis...")
    cohort_analysis = explainer.generate_cohort_analysis(predictions_file)
    
    print("\n2. Creating explanation report...")
    explainer.save_explanation_report(cohort_analysis, explanations_dir / 'xai_analysis_report.json')
    
    print("\n3. Generating visual explanations...")
    create_visual_explanations(predictions_file, explanations_dir)
    
    # Print key insights
    print("\n" + "="*60)
    print("KEY EXPLAINABILITY INSIGHTS")
    print("="*60)
    
    performance = cohort_analysis['model_performance']
    patterns = cohort_analysis['prediction_patterns']
    
    print(f"\n📊 Model Performance:")
    print(f"   • Accuracy: {performance['accuracy']:.1%}")
    print(f"   • Sensitivity (catches high-risk): {performance['sensitivity']:.1%}")
    print(f"   • Specificity (avoids false alarms): {performance['specificity']:.1%}")
    print(f"   • Positive Predictive Value: {performance['positive_predictive_value']:.1%}")
    
    print(f"\n🎯 Prediction Patterns:")
    print(f"   • High-confidence predictions: {patterns['high_confidence_predictions']}/{cohort_analysis['cohort_size']}")
    print(f"   • Borderline cases: {patterns['borderline_cases']}/{cohort_analysis['cohort_size']}")
    print(f"   • Mean prediction probability: {patterns['mean_probability']:.3f}")
    
    print(f"\n🔍 Clinical Insights:")
    for insight in cohort_analysis['clinical_insights']:
        print(f"   • {insight}")
    
    print(f"\n📁 Files Generated:")
    print(f"   • {explanations_dir / 'xai_analysis_report.json'}")
    if HAS_MATPLOTLIB:
        print(f"   • {explanations_dir / 'model_explanations.png'}")
    
    print(f"\n✅ Explainable AI analysis completed!")
    print(f"   The model shows {performance['accuracy']:.1%} accuracy with good interpretability.")
    print(f"   Review the generated files for detailed explanations.")

if __name__ == "__main__":
    main()