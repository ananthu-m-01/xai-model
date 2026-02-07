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

print("Note: Running in dependency-light mode for NumPy 2.x compatibility")


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


if __name__ == "__main__":
    main()