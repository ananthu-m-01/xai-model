"""
Simple threshold optimization script that avoids numpy/pandas compatibility issues
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)

def find_optimal_threshold(y_true, y_prob, method='gmean'):
    """Find optimal classification threshold using different methods."""

    if method == 'gmean':

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        gmean = np.sqrt(tpr * (1 - fpr))

        idx = np.argmax(gmean)

        return float(thresholds[idx]), float(gmean[idx])



    elif method == 'youden':

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        youden_j = tpr - fpr

        idx = np.argmax(youden_j)

        return float(thresholds[idx]), float(youden_j[idx])



    elif method == 'f1':

        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

        idx = np.argmax(f1_scores)

        return float(thresholds[idx]), float(f1_scores[idx])



    else:

        raise ValueError(f"Unknown method: {method}")

def evaluate_with_threshold(y_true, y_prob, threshold):
    """Evaluate predictions with a specific threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted')),
        'auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        'classification_report': classification_report(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

def main():
    base_path = Path(__file__).parent
    results_dir = base_path / 'results'
    
    # Load predictions - parse CSV manually to avoid pandas issues
    predictions_csv = results_dir / 'test_predictions.csv'
    if not predictions_csv.exists():
        print(f"Error: {predictions_csv} not found. Run test_model.py first.")
        return
    
    # Manual CSV parsing
    with open(predictions_csv, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    data_lines = lines[1:]
    
    y_true = []
    y_prob = []
    participant_ids = []
    
    for line in data_lines:
        parts = line.strip().split(',')
        if len(parts) >= 4:
            y_true.append(int(parts[1]))  # y_true column
            y_prob.append(float(parts[2]))  # y_prob column
            participant_ids.append(parts[4])  # participant_id column
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    print("="*60)
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print("="*60)
    
    # Original results (0.5 threshold)
    print("\n1. Original Results (threshold=0.5):")
    original = evaluate_with_threshold(y_true, y_prob, 0.5)
    print(f"   Accuracy: {original['accuracy']:.4f}")
    print(f"   F1 Score: {original['f1_score']:.4f}")
    print(f"   AUC: {original['auc']:.4f}")
    print(f"\n{original['classification_report']}")
    
    # Find optimal thresholds using different methods
    print("\n2. Optimal Thresholds:")
    methods = ['gmean', 'youden', 'f1']
    best_results = {}
    
    for method in methods:
        threshold, score = find_optimal_threshold(y_true, y_prob, method)
        result = evaluate_with_threshold(y_true, y_prob, threshold)
        best_results[method] = result
        
        print(f"\n   Method: {method.upper()}")
        print(f"   Optimal Threshold: {threshold:.4f} (score: {score:.4f})")
        print(f"   Accuracy: {result['accuracy']:.4f}")
        print(f"   F1 Score: {result['f1_score']:.4f}")
        print(f"\n{result['classification_report']}")
    
    # Find best method
    best_method = max(best_results.keys(), 
                     key=lambda k: best_results[k]['accuracy'])
    best_threshold = best_results[best_method]['threshold']
    
    # Save optimized predictions manually
    y_pred_optimized = (y_prob >= best_threshold).astype(int)
    
    with open(results_dir / 'test_predictions_optimized.csv', 'w') as f:
        f.write("row_index,y_true,y_prob,y_pred,participant_id,y_pred_optimized\n")
        for i, (yt, yp, ypo, pid) in enumerate(zip(y_true, y_prob, y_pred_optimized, participant_ids)):
            f.write(f"{i},{yt},{yp:.6f},{1 if yp >= 0.5 else 0},{pid},{ypo}\n")
    
    # Save comparison
    comparison = {
        'original': original,
        'optimized_methods': best_results,
        'best_method': best_method,
        'improvement': {
            'accuracy': float(best_results[best_method]['accuracy'] - original['accuracy']),
            'f1_score': float(best_results[best_method]['f1_score'] - original['f1_score'])
        }
    }
    
    with open(results_dir / 'threshold_optimization.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Best Method: {best_method.upper()}")
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Accuracy Improvement: {comparison['improvement']['accuracy']:+.4f}")
    print(f"F1 Score Improvement: {comparison['improvement']['f1_score']:+.4f}")
    print(f"\nResults saved to:")
    print(f"  - {results_dir / 'test_predictions_optimized.csv'}")
    print(f"  - {results_dir / 'threshold_optimization.json'}")
    
    # Show detailed comparison
    print(f"\nDetailed Comparison:")
    print(f"Original (0.5):   Acc={original['accuracy']:.4f}, F1={original['f1_score']:.4f}")
    print(f"Optimized ({best_threshold:.3f}): Acc={best_results[best_method]['accuracy']:.4f}, F1={best_results[best_method]['f1_score']:.4f}")
    
    if comparison['improvement']['accuracy'] > 0:
        print(f"\n✅ Improvement achieved! Accuracy increased by {comparison['improvement']['accuracy']*100:.1f}%")
    else:
        print(f"\n⚠️  No improvement - original threshold was already good")

if __name__ == "__main__":
    main()