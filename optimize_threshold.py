"""
Optimize classification threshold for better performance without retraining.
This version avoids heavy binary dependencies (pandas, scikit-learn) so it
remains compatible with environments running NumPy 2.x.
"""

import csv
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

HAS_SEABORN = False
try:
    import seaborn as sns  # type: ignore

    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    print("Note: seaborn not found, using matplotlib defaults")

HAS_MATPLOTLIB = False
try:
    import matplotlib.pyplot as _plt  # type: ignore

    HAS_MATPLOTLIB = True
except ImportError:
    _plt = None
    print("Warning: matplotlib not found, plotting will be disabled")


def _safe_divide(num: float, den: float) -> float:
    """Gracefully handle division by zero."""
    return float(num / den) if den else 0.0


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (tp, fp, tn, fn) for binary predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def _weighted_f1(tp: int, fp: int, tn: int, fn: int) -> float:
    """Compute weighted F1 score for binary classification."""
    support_pos = tp + fn
    support_neg = tn + fp
    total = support_pos + support_neg

    precision_pos = _safe_divide(tp, tp + fp)
    recall_pos = _safe_divide(tp, tp + fn)
    f1_pos = _safe_divide(2 * precision_pos * recall_pos, precision_pos + recall_pos)

    precision_neg = _safe_divide(tn, tn + fn)
    recall_neg = _safe_divide(tn, tn + fp)
    f1_neg = _safe_divide(2 * precision_neg * recall_neg, precision_neg + recall_neg)

    if total == 0:
        return 0.0
    return ((f1_pos * support_pos) + (f1_neg * support_neg)) / total


def _classification_report(tp: int, fp: int, tn: int, fn: int) -> str:
    """Return a compact text classification report for binary labels."""
    support_pos = tp + fn
    support_neg = tn + fp
    precision_pos = _safe_divide(tp, tp + fp)
    recall_pos = _safe_divide(tp, tp + fn)
    f1_pos = _safe_divide(2 * precision_pos * recall_pos, precision_pos + recall_pos)

    precision_neg = _safe_divide(tn, tn + fn)
    recall_neg = _safe_divide(tn, tn + fp)
    f1_neg = _safe_divide(2 * precision_neg * recall_neg, precision_neg + recall_neg)

    accuracy = _safe_divide(tp + tn, support_pos + support_neg)

    lines = [
        "Class  precision  recall  f1-score  support",
        f"0      {precision_neg:9.3f}  {recall_neg:6.3f}  {f1_neg:8.3f}  {support_neg:7d}",
        f"1      {precision_pos:9.3f}  {recall_pos:6.3f}  {f1_pos:8.3f}  {support_pos:7d}",
        f"\nAccuracy: {accuracy:.3f}",
    ]
    return "\n".join(lines)


def _confusion_matrix(tp: int, fp: int, tn: int, fn: int) -> list:
    """Return confusion matrix as nested Python lists (rows: actual 0,1)."""
    return [[tn, fp], [fn, tp]]


def manual_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC AUC manually to avoid SciPy/Sklearn dependencies."""
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    order = np.argsort(-y_scores)
    y_true_sorted = y_true[order]

    n_pos = np.sum(y_true_sorted == 1)
    n_neg = np.sum(y_true_sorted == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    rank_sum = 0.0
    for idx, label in enumerate(y_true_sorted, start=1):
        if label == 1:
            rank_sum += idx

    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _threshold_candidates(y_prob: np.ndarray) -> np.ndarray:
    """Generate candidate thresholds from probabilities plus midpoints."""
    scores = np.sort(np.unique(y_prob))
    if scores.size == 0:
        return np.array([0.5], dtype=float)

    midpoints = (scores[:-1] + scores[1:]) / 2.0 if scores.size > 1 else np.array([], dtype=float)
    candidates = np.concatenate(([0.0, 0.5, 1.0], scores, midpoints))
    candidates = np.clip(candidates, 0.0, 1.0)
    return np.unique(candidates)


def _precision_recall(tp: int, fp: int, tn: int, fn: int) -> Tuple[float, float]:
    """Return positive-class precision and recall."""
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return precision, recall


def find_optimal_threshold(y_true: Iterable[int], y_prob: Iterable[float], method: str = 'gmean') -> Tuple[float, float]:
    """Find optimal threshold based on specified metric."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    best_threshold = 0.5
    best_score = -np.inf

    for threshold in _threshold_candidates(y_prob):
        y_pred = (y_prob >= threshold).astype(int)
        tp, fp, tn, fn = _binary_counts(y_true, y_pred)
        precision, recall = _precision_recall(tp, fp, tn, fn)
        tpr = recall
        fpr = _safe_divide(fp, fp + tn)

        if method == 'gmean':
            score = float(np.sqrt(max(tpr * (1.0 - fpr), 0.0)))
        elif method == 'youden':
            score = float(tpr - fpr)
        elif method == 'f1':
            score = _safe_divide(2 * precision * recall, precision + recall)
        else:
            raise ValueError(f"Unknown method '{method}'. Valid options: 'gmean', 'youden', 'f1'")

        if np.isnan(score):
            continue

        if score > best_score or (np.isclose(score, best_score) and threshold < best_threshold):
            best_score = score
            best_threshold = threshold

    if best_score == -np.inf:
        return 0.5, 0.0
    return float(best_threshold), float(best_score)


def evaluate_with_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Evaluate predictions with a specific threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp, fp, tn, fn = _binary_counts(y_true, y_pred)

    accuracy = _safe_divide(tp + tn, len(y_true))
    weighted_f1 = _weighted_f1(tp, fp, tn, fn)
    auc_value = manual_auc(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'f1_score': float(weighted_f1),
        'auc': float(auc_value),
        'classification_report': _classification_report(tp, fp, tn, fn),
        'confusion_matrix': _confusion_matrix(tp, fp, tn, fn),
    }


def _curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate ROC and PR points for plotting."""
    roc_records = []
    pr_records = []

    for threshold in _threshold_candidates(y_prob):
        y_pred = (y_prob >= threshold).astype(int)
        tp, fp, tn, fn = _binary_counts(y_true, y_pred)
        precision, recall = _precision_recall(tp, fp, tn, fn)
        tpr = recall
        fpr = _safe_divide(fp, fp + tn)
        gmean = np.sqrt(max(tpr * (1.0 - fpr), 0.0))

        roc_records.append((fpr, tpr, threshold, gmean))
        pr_records.append((recall, precision))

    roc_records.sort(key=lambda x: (x[0], x[1]))
    pr_records.sort(key=lambda x: (x[0], -x[1]))

    fpr = np.array([r[0] for r in roc_records])
    tpr = np.array([r[1] for r in roc_records])
    thresholds = np.array([r[2] for r in roc_records])
    gmeans = np.array([r[3] for r in roc_records])
    recalls = np.array([r[0] for r in pr_records])
    precisions = np.array([r[1] for r in pr_records])

    return fpr, tpr, thresholds, recalls, precisions, gmeans


def plot_threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path, best_threshold: float) -> None:
    """Create visualizations for threshold analysis."""
    if not HAS_MATPLOTLIB or _plt is None:
        print("Matplotlib not available, skipping plots")
        return

    plt = _plt
    output_dir.mkdir(exist_ok=True, parents=True)

    fpr, tpr, thresholds, recalls, precisions, gmeans = _curve_points(y_true, y_prob)
    auc_value = manual_auc(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    best_idx = int(np.argmax(gmeans)) if gmeans.size else 0

    # ROC Curve
    plt.figure(figsize=(10, 6))
    if np.isfinite(auc_value):
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc_value:.3f})')
    else:
        plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    if thresholds.size:
        plt.scatter(fpr[best_idx], tpr[best_idx], c='red', s=100,
                    label=f'Optimal (threshold={thresholds[best_idx]:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Optimal Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Metrics vs Threshold
    thresholds_range = np.linspace(0.0, 1.0, 101)
    accuracies = []
    f1_scores = []
    for thr in thresholds_range:
        y_pred = (y_prob >= thr).astype(int)
        tp, fp, tn, fn = _binary_counts(y_true, y_pred)
        accuracies.append(_safe_divide(tp + tn, len(y_true)))
        f1_scores.append(_weighted_f1(tp, fp, tn, fn))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_range, accuracies, label='Accuracy', marker='o')
    plt.plot(thresholds_range, f1_scores, label='Weighted F1', marker='s')
    plt.axvline(x=best_threshold, color='r', linestyle='--',
                label=f'Optimal Threshold ({best_threshold:.3f})')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Classification Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'threshold_vs_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def main() -> None:
    base_path = Path(__file__).parent
    results_dir = base_path / 'results'

    predictions_csv = results_dir / 'test_predictions.csv'
    if not predictions_csv.exists():
        print(f"Error: {predictions_csv} not found. Run test_model.py first.")
        return

    records = []
    with open(predictions_csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    if not records:
        print(f"Error: {predictions_csv} is empty")
        return

    y_true = np.array([int(float(r['y_true'])) for r in records], dtype=np.int32)
    y_prob = np.array([float(r['y_prob']) for r in records], dtype=np.float32)

    print("=" * 60)
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print("=" * 60)

    print("\n1. Original Results (threshold=0.5):")
    original = evaluate_with_threshold(y_true, y_prob, 0.5)
    print(f"   Accuracy: {original['accuracy']:.4f}")
    print(f"   F1 Score: {original['f1_score']:.4f}")
    print(f"   AUC: {original['auc']:.4f}")
    print(f"\n{original['classification_report']}")

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

    best_method = max(best_results.keys(), key=lambda k: best_results[k]['accuracy'])
    best_threshold = best_results[best_method]['threshold']

    optimized_preds = (y_prob >= best_threshold).astype(int)
    optimized_path = results_dir / 'test_predictions_optimized.csv'

    with open(optimized_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['row_index', 'y_true', 'y_prob', 'y_pred_original', 'participant_id', 'y_pred_optimized']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, opt_pred in zip(records, optimized_preds):
            writer.writerow({
                'row_index': row.get('row_index', ''),
                'y_true': int(float(row['y_true'])),
                'y_prob': float(row['y_prob']),
                'y_pred_original': int(float(row.get('y_pred_original', row.get('y_pred', 0)))),
                'participant_id': row.get('participant_id', ''),
                'y_pred_optimized': int(opt_pred),
            })

    comparison = {
        'original': original,
        'optimized_methods': best_results,
        'best_method': best_method,
        'improvement': {
            'accuracy': float(best_results[best_method]['accuracy'] - original['accuracy']),
            'f1_score': float(best_results[best_method]['f1_score'] - original['f1_score']),
        },
    }

    with open(results_dir / 'threshold_optimization.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)

    print("\n3. Generating visualizations...")
    if HAS_MATPLOTLIB and _plt is not None:
        plot_threshold_analysis(y_true, y_prob, results_dir / 'threshold_plots', best_threshold)
        plots_msg = f"  - {results_dir / 'threshold_plots'}/ (visualizations)"
    else:
        plots_msg = "  - Plots skipped (matplotlib not available)"

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Best Method: {best_method.upper()}")
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Accuracy Improvement: {comparison['improvement']['accuracy']:+.4f}")
    print(f"F1 Score Improvement: {comparison['improvement']['f1_score']:+.4f}")
    print("\nResults saved to:")
    print(f"  - {optimized_path}")
    print(f"  - {results_dir / 'threshold_optimization.json'}")
    print(plots_msg)


if __name__ == "__main__":
    main()

def find_optimal_threshold(y_true, y_prob, method='gmean'):
   
    # Handle degenerate case: only one class present in y_true
    if len(np.unique(y_true)) < 2:
        # Cannot compute meaningful threshold; return default
        return 0.5, 0.0

    if method == 'gmean':
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        gmean = np.sqrt(tpr * (1 - fpr))
        idx = np.nanargmax(gmean)
        return float(thresholds[idx]), float(gmean[idx])
    
    elif method == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_j = tpr - fpr
        idx = np.nanargmax(youden_j)
        return float(thresholds[idx]), float(youden_j[idx])
    
    elif method == 'f1':
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        # precision and recall have length len(thresholds) + 1; align by excluding last element
        if len(thresholds) == 0:
            return 0.5, 0.0
        p = precision[:-1]
        r = recall[:-1]
        f1_scores = 2 * (p * r) / (p + r + 1e-10)
        idx = np.nanargmax(f1_scores)
        return float(thresholds[idx]), float(f1_scores[idx])

    else:
        raise ValueError(f"Unknown method '{method}'. Valid options: 'gmean', 'youden', 'f1'")

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

def plot_threshold_analysis(y_true, y_prob, output_dir):
    """Create visualizations for threshold analysis."""
    if not HAS_MATPLOTLIB or _plt is None:
        print("Matplotlib not available, skipping plots")
        return

    plt = _plt
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. ROC Curve with optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    gmean = np.sqrt(tpr * (1 - fpr))
    idx_gmean = np.argmax(gmean)

    auc_value = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    
    plt.figure(figsize=(10, 6))
    if np.isfinite(auc_value):
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc_value:.3f})')
    else:
        plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.scatter(fpr[idx_gmean], tpr[idx_gmean], c='red', s=100, 
                label=f'Optimal (threshold={thresholds[idx_gmean]:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Optimal Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Threshold vs Metrics
    thresholds_range = np.linspace(0.1, 0.9, 50)
    accuracies = []
    f1_scores = []
    
    for t in thresholds_range:
        y_pred = (y_prob >= t).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred, average='weighted'))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_range, accuracies, label='Accuracy', marker='o')
    plt.plot(thresholds_range, f1_scores, label='F1 Score', marker='s')
    plt.axvline(x=thresholds[idx_gmean], color='r', linestyle='--', 
                label=f'Optimal Threshold ({thresholds[idx_gmean]:.3f})')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Classification Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'threshold_vs_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    base_path = Path(__file__).parent
    results_dir = base_path / 'results'
    
    # Load predictions
    predictions_csv = results_dir / 'test_predictions.csv'
    if not predictions_csv.exists():
        print(f"Error: {predictions_csv} not found. Run test_model.py first.")
        return
    
    records = []
    with open(predictions_csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    if not records:
        print(f"Error: {predictions_csv} is empty")
        return

    y_true = np.array([int(float(r['y_true'])) for r in records], dtype=np.int32)
    y_prob = np.array([float(r['y_prob']) for r in records], dtype=np.float32)
    
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
    
    # Save optimized predictions
    best_method = max(best_results.keys(), 
                     key=lambda k: best_results[k]['accuracy'])
    best_threshold = best_results[best_method]['threshold']
    
    optimized_preds = (y_prob >= best_threshold).astype(int)

    original_pred_key = 'y_pred_original' if 'y_pred_original' in records[0] else 'y_pred'

    optimized_path = results_dir / 'test_predictions_optimized.csv'
    with open(optimized_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['row_index', 'y_true', 'y_prob', 'y_pred_original', 'participant_id', 'y_pred_optimized']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, opt_pred in zip(records, optimized_preds):
            writer.writerow({
                'row_index': row.get('row_index', ''),
                'y_true': int(float(row['y_true'])),
                'y_prob': float(row['y_prob']),
                'y_pred_original': int(float(row.get('y_pred_original', row.get('y_pred', 0)))),
                'participant_id': row.get('participant_id', ''),
                'y_pred_optimized': int(opt_pred)
            })
    
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
    
    with open(results_dir / 'threshold_optimization.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    
    # Create visualizations
    print("\n3. Generating visualizations...")
    if HAS_MATPLOTLIB:
        plot_threshold_analysis(y_true, y_prob, results_dir / 'threshold_plots')
        plots_msg = f"  - {results_dir / 'threshold_plots'}/ (visualizations)"
    else:
        plots_msg = "  - Plots skipped (matplotlib not available)"
    
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
    print(f"{plots_msg}")

if __name__ == "__main__":
    main()
