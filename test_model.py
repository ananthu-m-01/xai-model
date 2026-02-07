import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch

# Reuse your implementations
from hybrid_model import (
    Config, BrainDataset, MultiModalNet, evaluate, tune_threshold_roc_gmean
)

def build_splits(y, seed=Config.SEED):
    """Replicates the train/val/test split used in training."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr_val_idx, te_idx = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=y
    )
    # A second split exists in training only for train/val; for testing we only need te_idx
    return tr_val_idx, te_idx

def load_threshold(results_dir: Path) -> float:
    metrics_file = results_dir / "test_metrics_thresholded.json"
    if metrics_file.exists():
        try:
            data = json.loads(metrics_file.read_text())
            thr = float(data.get("threshold", 0.5))
            print(f"[info] Loaded threshold {thr:.4f} from {metrics_file}")
            return thr
        except Exception:
            pass
    print("[info] Using default threshold 0.5 (no saved threshold found)")
    return 0.5

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test split")
    parser.add_argument("--model-path", type=str, default="best_model_fused.pth", help="Path to model state_dict")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save outputs")
    parser.add_argument("--batch-size", type=int, default=Config.TEST_BATCH_SIZE, help="Batch size for inference")
    parser.add_argument("--recompute-threshold", action="store_true", help="Recompute threshold from validation set")
    args = parser.parse_args()

    base_path = Path(__file__).parent
    results_dir = (base_path / args.results_dir)
    results_dir.mkdir(exist_ok=True)

    # Load data
    df = pd.read_csv(base_path / "participants_with_labels.csv")

    # Reproduce the label used in training
    df['risk_label'] = (
        (df['dementia_history_parents'].astype(float).fillna(0) > 0) |
        (df.get('CVLT_7', pd.Series([13.5]*len(df))).astype(float).fillna(13.5) < 13.5)
    ).astype(int)

    # Feature placeholders (replace with real features if available)
    X_eeg = np.random.randn(len(df), 16).astype(np.float32)
    X_fmri = np.random.randn(len(df), 32).astype(np.float32)
    text_df = df[Config.TEXT_FEATURES].copy()
    y = df['risk_label'].values.astype(np.int64)

    # Build the same test split as training
    _, te_idx = build_splits(y, seed=Config.SEED)

    X_te_eeg = X_eeg[te_idx]
    X_te_fmri = X_fmri[te_idx]
    text_te = text_df.iloc[te_idx]
    y_te = y[te_idx]

    # Dataset & loader
    test_ds = BrainDataset(X_te_eeg, X_te_fmri, text_te, y_te, augment=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=(Config.DEVICE.type == "cuda"),
        persistent_workers=True
    )

    # Model
    model = MultiModalNet(eeg_dim=X_te_eeg.shape[1], fmri_dim=X_te_fmri.shape[1]).to(Config.DEVICE)

    # Load weights
    model_path = base_path / args.model_path
    state = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[info] Loaded model weights from {model_path}")

    # Evaluate to get probabilities
    eval_out = evaluate(model, test_loader, criterion=None, return_probs=True)

    # Threshold
    if args.recompute_threshold:
        # Recompute threshold using ROC G-Mean on test probs and labels (for analysis only)
        thr, gmean = tune_threshold_roc_gmean(eval_out['trues'], eval_out['probs'])
        print(f"[info] Recomputed threshold from test: {thr:.4f} (G-Mean={gmean:.4f})")
    else:
        thr = load_threshold(results_dir=results_dir)

    # Metrics with threshold
    y_prob = np.array(eval_out['probs'])
    y_true = np.array(eval_out['trues'])
    y_pred = (y_prob >= thr).astype(np.int64)

    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score, average_precision_score
    )
    metrics = {
        "accuracy": float(np.mean(y_pred == y_true)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "classification_report": classification_report(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": float(thr)
    }

    print(json.dumps(metrics, indent=2))

    # Save metrics and predictions
    (results_dir / "inference_metrics.json").write_text(json.dumps(metrics, indent=2))

    out_df = pd.DataFrame({
        "row_index": te_idx,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred
    })
    # Add participant_id if present in CSV
    pid_col = "participant_id" if "participant_id" in df.columns else None
    if pid_col:
        out_df[pid_col] = df.iloc[te_idx][pid_col].values

    out_df.to_csv(results_dir / "test_predictions.csv", index=False)
    print(f"[info] Saved predictions to {results_dir / 'test_predictions.csv'}")

if __name__ == "__main__":
    main()