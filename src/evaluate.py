"""
Comprehensive evaluation module for the MNIST CNN model on the untouched test set.
Generates metrics, classification reports, confusion matrices, training curves,
and error analysis visualizations.
"""

from pathlib import Path
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend for server/script execution
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import get_data_loaders, load_mnist_datasets
from src.model import MNISTCNN
from src.train import load_checkpoint
from src.utils import ensure_dir, get_device, load_config, set_seed


def evaluate_test_set(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Perform a complete forward pass over the untouched test set.

    Calculates test loss, accuracy, precision, recall, F1 (macro and weighted),
    per-class metrics, confusion matrix, and misclassified sample records.
    """
    model.eval()
    running_loss = 0.0
    all_targets: List[int] = []
    all_predictions: List[int] = []
    all_confidences: List[float] = []
    all_images: List[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images_dev = images.to(device, non_blocking=True)
            labels_dev = labels.to(device, non_blocking=True)

            logits = model(images_dev)
            loss = criterion(logits, labels_dev)

            running_loss += loss.item() * images.size(0)

            # Softmax computed strictly during evaluation for probability interpretation
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_targets.extend(labels.tolist())
            all_predictions.extend(preds.cpu().tolist())
            all_confidences.extend(confs.cpu().tolist())
            all_images.extend(images.cpu())

    total_samples = len(all_targets)
    test_loss = running_loss / total_samples

    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    confs = np.array(all_confidences)

    correct_mask = (y_true == y_pred)
    correct_count = int(np.sum(correct_mask))
    incorrect_count = int(total_samples - correct_count)
    accuracy = correct_count / total_samples

    # Summary metrics
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Scikit-learn detailed classification report
    clf_report = classification_report(
        y_true, y_pred, output_dict=True, digits=4, zero_division=0
    )

    # 10x10 Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    # Per-class summary extraction
    per_class_metrics = {}
    for digit in range(10):
        digit_key = str(digit)
        if digit_key in clf_report:
            per_class_metrics[digit] = {
                "precision": round(clf_report[digit_key]["precision"], 4),
                "recall": round(clf_report[digit_key]["recall"], 4),
                "f1_score": round(clf_report[digit_key]["f1-score"], 4),
                "support": int(clf_report[digit_key]["support"]),
            }

    # Identify most common confusion pairs (where actual != predicted)
    confusion_pairs = []
    for actual in range(10):
        for pred in range(10):
            if actual != pred and cm[actual, pred] > 0:
                confusion_pairs.append({
                    "actual": actual,
                    "predicted": pred,
                    "count": int(cm[actual, pred]),
                })
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)

    # Easiest and hardest digits by F1-score
    sorted_by_f1 = sorted(per_class_metrics.items(), key=lambda x: x[1]["f1_score"])
    hardest_digit = sorted_by_f1[0][0]
    easiest_digit = sorted_by_f1[-1][0]

    # Collect misclassified samples metadata and tensors
    incorrect_indices = np.where(~correct_mask)[0]
    misclassified_records = []
    for idx in incorrect_indices:
        misclassified_records.append({
            "index": int(idx),
            "actual": int(y_true[idx]),
            "predicted": int(y_pred[idx]),
            "confidence": float(confs[idx]),
            "tensor": all_images[idx],
        })

    # Sort misclassifications by model confidence (most confident errors first)
    misclassified_records.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(accuracy, 6),
        "total_samples": total_samples,
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_pairs": confusion_pairs,
        "easiest_digit": easiest_digit,
        "hardest_digit": hardest_digit,
        "clf_report": clf_report,
        "misclassified_records": misclassified_records,
    }


def plot_confusion_matrices(
    cm: np.ndarray,
    output_raw_path: Path,
    output_norm_path: Path,
) -> None:
    """Generate both raw-count and normalized confusion matrix heatmaps."""
    digits = [str(i) for i in range(10)]

    # 1. Raw Count Heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=digits,
        yticklabels=digits,
        cbar=True,
        linewidths=0.5,
    )
    plt.title("MNIST CNN Confusion Matrix (Test Set Counts)", fontsize=14, pad=12)
    plt.xlabel("Predicted Digit", fontsize=12)
    plt.ylabel("Actual Digit", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_raw_path, dpi=300)
    plt.close()

    # 2. Normalized Heatmap (row-normalized by actual class count)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=digits,
        yticklabels=digits,
        cbar=True,
        linewidths=0.5,
    )
    plt.title("MNIST CNN Normalized Confusion Matrix (Test Set Proportions)", fontsize=14, pad=12)
    plt.xlabel("Predicted Digit", fontsize=12)
    plt.ylabel("Actual Digit", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_norm_path, dpi=300)
    plt.close()


def plot_training_curves(
    history_data: Dict[str, Any],
    output_path: Path,
) -> None:
    """Plot dual-panel learning curves (Loss and Accuracy vs Epochs) from training history."""
    hist = history_data.get("history", history_data)
    epochs = hist["epochs"]
    train_loss = hist["train_loss"]
    val_loss = hist["val_loss"]
    train_acc = [acc * 100 for acc in hist["train_acc"]]
    val_acc = [acc * 100 for acc in hist["val_acc"]]
    best_epoch = history_data.get("best_epoch", int(np.argmin(val_loss)) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs, train_loss, "o-", label="Training Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, val_loss, "s-", label="Validation Loss", color="#ff7f0e", linewidth=2)
    ax1.axvline(best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Checkpoint (Epoch {best_epoch})")
    ax1.set_title("Cross-Entropy Loss vs. Epoch", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # Accuracy Curve
    ax2.plot(epochs, train_acc, "o-", label="Training Accuracy", color="#1f77b4", linewidth=2)
    ax2.plot(epochs, val_acc, "s-", label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    ax2.axvline(best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Checkpoint (Epoch {best_epoch})")
    ax2.set_title("Classification Accuracy vs. Epoch", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_error_analysis(
    misclassified_records: List[Dict[str, Any]],
    output_path: Path,
    num_samples: int = 16,
) -> None:
    """
    Generate a 4x4 visual diagnostic grid of misclassified test digits,
    showing actual label, predicted label, and model prediction confidence.
    """
    total_errors = len(misclassified_records)
    samples_to_show = min(total_errors, num_samples)

    if samples_to_show == 0:
        return

    # Select representative high-confidence errors
    selected = misclassified_records[:samples_to_show]

    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(rows * cols):
        ax = axes[i]
        if i < len(selected):
            record = selected[i]
            tensor = record["tensor"]  # Shape (1, 28, 28)
            actual = record["actual"]
            predicted = record["predicted"]
            conf = record["confidence"]

            # Reverse MNIST normalization for clean visualization: (pixel * std) + mean
            img_np = tensor.squeeze().numpy() * 0.3081 + 0.1307
            img_np = np.clip(img_np, 0.0, 1.0)

            ax.imshow(img_np, cmap="gray")
            ax.set_title(
                f"True: {actual} | Pred: {predicted}\nConf: {conf * 100:.1f}%",
                fontsize=10,
                color="#d62728",
                fontweight="bold",
            )
        ax.axis("off")

    plt.suptitle(
        f"MNIST CNN Error Analysis (Top {samples_to_show} Misclassifications from {total_errors} Total Errors)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_evaluation(
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Main evaluation pipeline orchestrating test set benchmark,
    figure generation, and results saving.
    """
    t0 = time.time()

    if config is None:
        config = load_config()

    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)
    device = get_device()

    # Paths
    paths_cfg = config.get("paths", {})
    checkpoint_path = paths_cfg.get("best_model_path", "models/best_model.pth")
    history_path = Path(paths_cfg.get("training_history_path", "models/training_history.json"))
    figures_dir = ensure_dir(paths_cfg.get("figures_dir", "reports/figures"))
    reports_dir = ensure_dir(paths_cfg.get("reports_dir", "reports"))

    eval_json_path = Path(paths_cfg.get("evaluation_results_path", "reports/evaluation_results.json"))
    clf_json_path = Path(paths_cfg.get("classification_report_path", "reports/classification_report.json"))
    cm_path = Path(paths_cfg.get("confusion_matrix_path", "reports/figures/confusion_matrix.png"))
    cm_norm_path = Path(paths_cfg.get("confusion_matrix_norm_path", "reports/figures/confusion_matrix_normalized.png"))
    curves_path = Path(paths_cfg.get("training_curves_path", "reports/figures/training_curves.png"))
    error_path = Path(paths_cfg.get("error_analysis_path", "reports/figures/error_analysis.png"))

    if verbose:
        print("=" * 78)
        print("Starting MNIST CNN Test-Set Evaluation & Benchmarking")
        print("=" * 78)
        print(f"Device:                 {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
        print(f"Model Checkpoint:       {checkpoint_path}")
        print("Test Set:               Canonical MNIST (10,000 samples, unaugmented)")
        print("=" * 78)

    # 1. Load trained model
    model, checkpoint_dict = load_checkpoint(checkpoint_path, device=device)
    criterion = nn.CrossEntropyLoss()

    # 2. Load untouched test set
    _, _, test_loader = get_data_loaders(config)

    # 3. Perform test evaluation
    results = evaluate_test_set(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    eval_duration = time.time() - t0

    # 4. Generate visualizations
    cm_array = np.array(results["confusion_matrix"])
    plot_confusion_matrices(cm_array, cm_path, cm_norm_path)

    if history_path.is_file():
        with open(history_path, "r", encoding="utf-8") as f:
            hist_data = json.load(f)
        plot_training_curves(hist_data, curves_path)

    plot_error_analysis(results["misclassified_records"], error_path, num_samples=16)

    # 5. Save machine-readable reports
    # Strip non-serializable tensors before JSON export
    serializable_results = {k: v for k, v in results.items() if k not in ["misclassified_records", "clf_report"]}
    serializable_results.update({
        "checkpoint_used": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_dict.get("epoch"),
        "evaluation_device": device.type,
        "evaluation_runtime_seconds": round(eval_duration, 2),
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)

    with open(clf_json_path, "w", encoding="utf-8") as f:
        json.dump(results["clf_report"], f, indent=2)

    if verbose:
        print(f"\nTest Set Performance Metrics ({results['total_samples']:,} samples):")
        print(f"  - Test Loss:              {results['test_loss']:.4f}")
        print(f"  - Test Accuracy:          {results['test_accuracy'] * 100:.2f}% ({results['correct_predictions']:,} / {results['total_samples']:,})")
        print(f"  - Total Misclassifications: {results['incorrect_predictions']}")
        print(f"  - Macro Precision:        {results['macro_precision']:.4f}")
        print(f"  - Macro Recall:           {results['macro_recall']:.4f}")
        print(f"  - Macro F1-Score:         {results['macro_f1']:.4f}")
        print(f"  - Weighted F1-Score:      {results['weighted_f1']:.4f}")

        print("\nPer-Class Classification Report:")
        print(f"{'Digit':<7}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Support'}")
        print("-" * 52)
        for d in range(10):
            m = results["per_class_metrics"][d]
            print(f"{d:<7}{m['precision']:<12.4f}{m['recall']:<12.4f}{m['f1_score']:<12.4f}{m['support']}")

        print("\nDiagnostic Summary:")
        print(f"  - Easiest Digit (Highest F1): Digit {results['easiest_digit']} (F1 = {results['per_class_metrics'][results['easiest_digit']]['f1_score']:.4f})")
        print(f"  - Hardest Digit (Lowest F1):  Digit {results['hardest_digit']} (F1 = {results['per_class_metrics'][results['hardest_digit']]['f1_score']:.4f})")
        
        top_confusion = results["confusion_pairs"][:5]
        print("\nTop Confusion Pairs (Actual -> Predicted):")
        for pair in top_confusion:
            print(f"  - Actual {pair['actual']} misclassified as {pair['predicted']}: {pair['count']} instances")

        print("\nGenerated Visualizations & Reports:")
        print(f"  - Confusion Matrix (Counts):     {cm_path}")
        print(f"  - Confusion Matrix (Normalized): {cm_norm_path}")
        print(f"  - Training Curves:               {curves_path}")
        print(f"  - Error Analysis (16 Examples):  {error_path}")
        print(f"  - Results JSON:                  {eval_json_path}")
        print(f"  - Classification Report JSON:    {clf_json_path}")
        print("=" * 78)

    return serializable_results


if __name__ == "__main__":
    run_evaluation()
