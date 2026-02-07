"""
Model Selection Guide for Enhanced Multi-Modal Training
Helps you choose the best Hugging Face model for your brain imaging task
"""

from model_huggingface import AdvancedConfig
import torch

def analyze_available_models():
    """Analyze available models and provide recommendations."""
    
    print("🧠 HUGGING FACE MODEL SELECTION GUIDE")
    print("="*80)
    
    print("\n📊 AVAILABLE MODELS:")
    print("-"*60)
    
    for key, model_path in AdvancedConfig.AVAILABLE_MODELS.items():
        config = AdvancedConfig.MODEL_CONFIGS[key]
        print(f"\n{key.upper()}: {model_path}")
        print(f"  • Type: {config['model_type'].upper()}")
        print(f"  • Hidden Dimension: {config['hidden_dim']}")
        print(f"  • Max Length: {config['max_length']}")
        
        # Add recommendations
        if 'biomedbert' in key or 'clinical' in key:
            print(f"  • Domain: 🏥 MEDICAL SPECIALIST")
            print(f"  • Best for: Medical text, clinical reports, health data")
        elif 'scibert' in key:
            print(f"  • Domain: 🔬 SCIENTIFIC SPECIALIST")
            print(f"  • Best for: Research papers, scientific literature")
        elif 'roberta' in key:
            print(f"  • Domain: 💪 GENERAL PURPOSE (STRONG)")
            print(f"  • Best for: General text understanding, robust performance")
        elif 'deberta' in key:
            print(f"  • Domain: 🎯 GENERAL PURPOSE (LATEST)")
            print(f"  • Best for: State-of-the-art general text understanding")
        elif 'electra' in key:
            print(f"  • Domain: ⚡ EFFICIENT (FAST)")
            print(f"  • Best for: Fast training, efficient inference")
        elif 'distil' in key:
            print(f"  • Domain: 🏃 LIGHTWEIGHT")
            print(f"  • Best for: Limited GPU memory, fast experiments")
        elif 'large' in key:
            print(f"  • Domain: 🚀 HIGH PERFORMANCE")
            print(f"  • Best for: Maximum accuracy, sufficient GPU memory")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("-"*60)
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"📺 Your GPU Memory: {gpu_mem:.1f} GB")
        
        if gpu_mem >= 16:
            recommended = "biomedbert_large or roberta_large"
            print(f"✅ RECOMMENDED: {recommended}")
            print("   Your GPU can handle large models for maximum performance!")
        elif gpu_mem >= 8:
            recommended = "biomedbert or clinicalbert"
            print(f"✅ RECOMMENDED: {recommended}")
            print("   Perfect balance of medical knowledge and performance!")
        else:
            recommended = "distilbert or distilroberta"
            print(f"✅ RECOMMENDED: {recommended}")
            print("   Efficient models that fit your GPU memory constraints!")
    else:
        print("⚠️  CPU Mode: Use distilbert for faster training")
        recommended = "distilbert"
    
    print(f"\n🏆 TOP CHOICES FOR MEDICAL AI:")
    print("1. 'biomedbert' - Best medical understanding")
    print("2. 'clinicalbert' - Clinical text specialist") 
    print("3. 'roberta' - Strong general performance")
    print("4. 'deberta' - Latest transformer technology")
    print("5. 'distilbert' - Fast and reliable fallback")
    
    print(f"\n⚙️ TO CHANGE MODEL:")
    print("Edit model_huggingface.py and change:")
    print("  AdvancedConfig.CHOSEN_MODEL = 'your_choice'")
    print(f"Current setting: '{AdvancedConfig.CHOSEN_MODEL}'")
    
    print(f"\n🚀 PERFORMANCE EXPECTATIONS:")
    print("• Medical models (biomedbert, clinicalbert): 90-95% accuracy potential")
    print("• General models (roberta, deberta): 85-92% accuracy potential") 
    print("• Efficient models (distilbert, electra): 82-88% accuracy potential")
    print("• Large models: +2-5% accuracy boost but 2-3x slower training")
    
    print(f"\n💡 TIPS:")
    print("• Start with 'biomedbert' - best for medical text")
    print("• If memory issues, try 'distilbert'")
    print("• For maximum performance, use 'biomedbert_large'")
    print("• Compare 2-3 models to find the best for your data")

if __name__ == "__main__":
    analyze_available_models()