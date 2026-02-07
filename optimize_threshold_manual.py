"""
Manual threshold optimization without dependencies (numpy/pandas/sklearn issues)
"""
import json
import math
from pathlib import Path

def manual_roc_curve(y_true, y_prob):
    """Manual implementation of ROC curve calculation."""
    # Sort by probability in descending order
    sorted_indices = sorted(range(len(y_prob)), key=lambda i: y_prob[i], reverse=True)
    
    thresholds = []
    tpr_values = []  # True Positive Rate
    fpr_values = []  # False Positive Rate
    
    # Count positives and negatives
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    
    # Add point (0, 0)
    thresholds.append(y_prob[sorted_indices[0]] + 0.1)
    tpr_values.append(0)
    fpr_values.append(0)
    
    tp = 0
    fp = 0
    
    for i in sorted_indices:
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        
        thresholds.append(y_prob[i])
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    # Add point (1, 1)
    thresholds.append(y_prob[sorted_indices[-1]] - 0.1)
    tpr_values.append(1)
    fpr_values.append(1)
    
    return fpr_values, tpr_values, thresholds

def find_optimal_threshold_manual(y_true, y_prob):
    """Find optimal threshold using G-Mean manually."""
    fpr, tpr, thresholds = manual_roc_curve(y_true, y_prob)
    
    best_threshold = 0.5
    best_gmean = 0
    
    for i in range(len(thresholds)):
        gmean = math.sqrt(tpr[i] * (1 - fpr[i]))
        if gmean > best_gmean:
            best_gmean = gmean
            best_threshold = thresholds[i]
    
    return best_threshold, best_gmean

def evaluate_manual(y_true, y_prob, threshold):
    """Manual evaluation without sklearn."""
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    
    # Calculate metrics manually
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    
    precision_class0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_class0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_class0 = 2 * (precision_class0 * recall_class0) / (precision_class0 + recall_class0) if (precision_class0 + recall_class0) > 0 else 0
    
    precision_class1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_class1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_class1 = 2 * (precision_class1 * recall_class1) / (precision_class1 + recall_class1) if (precision_class1 + recall_class1) > 0 else 0
    
    # Weighted F1
    support_class0 = tn + fp
    support_class1 = tp + fn
    total_support = support_class0 + support_class1
    
    if total_support > 0:
        f1_weighted = (f1_class0 * support_class0 + f1_class1 * support_class1) / total_support
    else:
        f1_weighted = 0
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'f1_score': f1_weighted,
        'precision_class0': precision_class0,
        'recall_class0': recall_class0,
        'precision_class1': precision_class1,
        'recall_class1': recall_class1,
        'confusion_matrix': [[tn, fp], [fn, tp]],
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }

def format_classification_report(result):
    """Format classification report manually."""
    report = f"""              precision    recall  f1-score   support

           0       {result['precision_class0']:.2f}      {result['recall_class0']:.2f}      {2*(result['precision_class0']*result['recall_class0'])/(result['precision_class0']+result['recall_class0']) if (result['precision_class0']+result['recall_class0'])>0 else 0:.2f}         {result['tn']+result['fp']}
           1       {result['precision_class1']:.2f}      {result['recall_class1']:.2f}      {2*(result['precision_class1']*result['recall_class1'])/(result['precision_class1']+result['recall_class1']) if (result['precision_class1']+result['recall_class1'])>0 else 0:.2f}        {result['tp']+result['fn']}

    accuracy                           {result['accuracy']:.2f}        {result['tp']+result['fp']+result['tn']+result['fn']}
   macro avg       {(result['precision_class0']+result['precision_class1'])/2:.2f}      {(result['recall_class0']+result['recall_class1'])/2:.2f}      {result['f1_score']:.2f}        {result['tp']+result['fp']+result['tn']+result['fn']}
weighted avg       {result['precision_class1']:.2f}      {result['recall_class1']:.2f}      {result['f1_score']:.2f}        {result['tp']+result['fp']+result['tn']+result['fn']}"""
    return report

def main():
    base_path = Path(__file__).parent
    results_dir = base_path / 'results'
    
    # Load predictions manually
    predictions_csv = results_dir / 'test_predictions.csv'
    if not predictions_csv.exists():
        print(f"Error: {predictions_csv} not found. Run test_model.py first.")
        return
    
    print("Loading predictions from CSV...")
    with open(predictions_csv, 'r') as f:
        lines = f.readlines()
    
    # Parse CSV manually
    data_lines = lines[1:]  # Skip header
    
    y_true = []
    y_prob = []
    participant_ids = []
    
    for line in data_lines:
        parts = line.strip().split(',')
        if len(parts) >= 4:
            y_true.append(int(parts[1]))  # y_true column
            y_prob.append(float(parts[2]))  # y_prob column
            participant_ids.append(parts[4])  # participant_id column
    
    print(f"Loaded {len(y_true)} predictions")
    
    print("="*60)
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print("="*60)
    
    # Original results (0.5 threshold)
    print("\n1. Original Results (threshold=0.5):")
    original = evaluate_manual(y_true, y_prob, 0.5)
    print(f"   Accuracy: {original['accuracy']:.4f}")
    print(f"   F1 Score: {original['f1_score']:.4f}")
    print(f"   Confusion Matrix: {original['confusion_matrix']}")
    print(format_classification_report(original))
    
    # Find optimal threshold
    print("\n2. Finding Optimal Threshold...")
    best_threshold, best_gmean = find_optimal_threshold_manual(y_true, y_prob)
    
    print(f"\n3. Optimal Threshold Results:")
    optimized = evaluate_manual(y_true, y_prob, best_threshold)
    print(f"   Optimal Threshold: {best_threshold:.4f} (G-Mean: {best_gmean:.4f})")
    print(f"   Accuracy: {optimized['accuracy']:.4f}")
    print(f"   F1 Score: {optimized['f1_score']:.4f}")
    print(f"   Confusion Matrix: {optimized['confusion_matrix']}")
    print(format_classification_report(optimized))
    
    # Save optimized predictions
    y_pred_optimized = [1 if p >= best_threshold else 0 for p in y_prob]
    y_pred_original = [1 if p >= 0.5 else 0 for p in y_prob]
    
    with open(results_dir / 'test_predictions_optimized.csv', 'w') as f:
        f.write("row_index,y_true,y_prob,y_pred_original,participant_id,y_pred_optimized\n")
        for i, (yt, yp, ypo, ypo_opt, pid) in enumerate(zip(y_true, y_prob, y_pred_original, y_pred_optimized, participant_ids)):
            f.write(f"{i},{yt},{yp:.6f},{ypo},{pid},{ypo_opt}\n")
    
    # Save comparison
    comparison = {
        'original': {
            'threshold': 0.5,
            'accuracy': original['accuracy'],
            'f1_score': original['f1_score'],
            'confusion_matrix': original['confusion_matrix']
        },
        'optimized': {
            'threshold': best_threshold,
            'accuracy': optimized['accuracy'],
            'f1_score': optimized['f1_score'], 
            'confusion_matrix': optimized['confusion_matrix'],
            'gmean_score': best_gmean
        },
        'improvement': {
            'accuracy': optimized['accuracy'] - original['accuracy'],
            'f1_score': optimized['f1_score'] - original['f1_score']
        }
    }
    
    with open(results_dir / 'threshold_optimization.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Original Threshold: 0.5000")
    print(f"Optimal Threshold:  {best_threshold:.4f}")
    print(f"Accuracy Improvement: {comparison['improvement']['accuracy']:+.4f}")
    print(f"F1 Score Improvement: {comparison['improvement']['f1_score']:+.4f}")
    
    print(f"\nDetailed Comparison:")
    print(f"Original (0.5):     Acc={original['accuracy']:.4f}, F1={original['f1_score']:.4f}")
    print(f"Optimized ({best_threshold:.3f}):  Acc={optimized['accuracy']:.4f}, F1={optimized['f1_score']:.4f}")
    
    print(f"\nPrediction Changes:")
    changes = sum(1 for i in range(len(y_pred_original)) if y_pred_original[i] != y_pred_optimized[i])
    print(f"  {changes} out of {len(y_true)} predictions changed")
    
    if comparison['improvement']['accuracy'] > 0:
        print(f"\n✅ SUCCESS! Accuracy improved by {comparison['improvement']['accuracy']*100:.1f}%")
    elif comparison['improvement']['accuracy'] == 0:
        print(f"\n⚠️  No change in accuracy")
    else:
        print(f"\n❌ Accuracy decreased by {abs(comparison['improvement']['accuracy'])*100:.1f}%")
    
    print(f"\nResults saved to:")
    print(f"  - {results_dir / 'test_predictions_optimized.csv'}")
    print(f"  - {results_dir / 'threshold_optimization.json'}")

if __name__ == "__main__":
    main()